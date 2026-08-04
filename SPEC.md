# Voxelium — roadmap

A from-scratch C++20 / OpenGL 4.6 (core profile, DSA everywhere) voxel engine. Not a
Minecraft clone — a good looking, heavily optimized voxel renderer.

The original M1–M12 milestone spec is finished; it lives in git history, with per-milestone
notes in `docs/M1..M12`. This file is the forward roadmap.

---

## Tier 0 — Broken / half-built

All closed.

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Teardown heap corruption | done\* | Did NOT reproduce in 52 runs (40 Release + 12 with `_GLIBCXX_ASSERTIONS`). Ruled out the palette straddle math, the arena free-list, and both chunk-meta writes. No libasan in this toolchain. Fixed a real data race on chunk light publication found while looking. Left open as unreproducible. |
| 2 | Wire up world persistence | done | **Was not dead code.** Already fully wired: load on gen, save on evict, `saveDirty()` at teardown, round-trip smoke test. Verified, nothing to do. |
| 3 | Cross-LOD seam skirts | done | Mesher takes a coarse-neighbour face mask and leaves those pads air. The mask is deliberately pessimistic (+2 chunks) so a stale one over-skirts instead of leaving a hole — that avoids needing a remesh cascade. +0.29% quads. |
| 4 | Teleport re-LOD hitch | done | Profiled first: the 1004 ms freeze was the arena doing `glFinish` + full rebuild. Now reclaims the fence ring before hard-resetting. Also killed a quadratic seed-queue deferral and two budgets that only checked their clock between large slices. Worst main-thread cost 6.8–11.5 ms, no hard resets. |
| 5 | Settings don't persist | done | `saves/settings.cfg`, flat key=value, clamped on load. `--smoke` deliberately ignores it so the reference image can't drift. |

Found and fixed along the way, not on the original list:

- **`kMaxChunks` was 8192** while the *default* radius 16 produces ~8700 meshed chunks.
  Everything past the cap got no meta entry — not drawn, and not in the shadow cull. This
  was the "some chunks have no shadows" and "chunks glitch out while moving". Now 32768,
  and it logs on overflow instead of truncating silently.
- **No seam skirts on water** — the skirt's air pad is shared by both mesh layers, so water
  grew vertical walls at chunk borders across every lake. Self-inflicted, from #3.
- **Stars were streaks, the moon was a small sun** — the star field folded a 3D direction
  into 2D via `dir.xy + dir.z`, so whole families of directions shared a cell. Now
  cube-face parameterised round dots; the moon is a sphere with maria and limb darkening.
- **Terraced ground read as corduroy** — `kShade` contrast, AO depth, and the actual cause:
  `grass_side` was mostly dirt, so risers were brown against green tops.
- **Incremental light seeding** — a column was one uninterruptible 5–10 ms unit.
- **Crosshair colour is configurable**, which also removed the driver shader recompile
  caused by `GL_ONE_MINUS_DST_COLOR` blending into an sRGB target.

---

## Tier 1 — Performance

| # | Item | Impact | Status |
|---|---|---|---|
| 6 | HiZ occlusion culling — depth pyramid from last frame, test chunk AABBs in `cull.comp`. Complements the existing flood-fill cave cull | High | todo |
| 7 | Render-scale + upscale (FSR-style) — render at 70%, upscale in `final.frag` | High | todo |
| 8 | Persistent-mapped SSBO uploads | High | **already done** — the quad arena is already `GL_MAP_PERSISTENT_BIT` with a 3-frame fence ring and a free-list sub-allocator. Nothing to do. |
| 9 | Shadow-pass LOD bias — render cascades one LOD coarser | High | todo |
| 10 | Shadow cascade caching — cascades 1–2 re-render every N frames | Med | todo |
| 11 | GPU light propagation — flood-fill in a compute shader | Med | todo |
| 12 | Sub-chunk (meshlet) culling — cull 8³ groups, not whole 32³ chunks | Med | todo |
| 13 | Arena defrag scheduling — amortize compaction instead of on-demand | Med | todo |
| 14 | Depth pre-pass — cuts overdraw for the heavy chunk fragment shader | Low–Med | todo |
| 15 | Async shader compile + pipeline cache | Low | todo |
| 16 | Chunk unload → disk instead of re-gen | Med | todo |

---

## Tier 2 — World & simulation

| # | Item | Status |
|---|---|---|
| 17 | Flowing water (fluid sim, no lava) | todo |
| 18 | Colored / RGB block light — needs the 4+4 light nibble widened | todo |
| 19 | Particles — block break, splash, footstep puffs | todo |
| 20 | Falling sand / gravel | todo |
| 21 | Sound engine + `sounds/` dir you populate (none bundled) | done |
| 22 | Exponential terrain height (2^noise) | todo |
| 23 | Better caves — worm + cheese caves | todo |
| 24 | Ore generation | todo |
| 25 | Foliage billboards — grass tufts, flowers | todo |
| 26 | Block break progress + animation | todo |
| 27 | Cubic chunks / no height limit — structural fork, `kWorldChunksY = 8` is hardcoded | todo |

---

## Tier 3 — Graphics polish

| # | Item | Status |
|---|---|---|
| 28 | Auto-exposure / eye adaptation — cave↔surface transitions blow out | todo |
| 29 | Weather — rain/snow particles + wet-surface darkening | todo |
| 30 | Water caustics + underwater god rays | todo |
| 31 | Volumetric raymarched clouds — replace the 2D cloud layer | todo |
| 32 | Cloud shadows on terrain | todo |
| 33 | Color grading LUT + subtle vignette/grain | todo |
| 34 | Sun disc + lens flare | todo |
| 35 | Screen-space contact shadows — fills PCSS's near-contact gap | todo |
| 36 | Proper atmosphere LUT (Hillaire) instead of the analytic sky | todo |
| 37 | Leaf subsurface scattering | todo |
| 38 | Normal maps / parallax on blocks — needs texture authoring | todo |
| 39 | Emissive bloom tuning for glow blocks | todo |

---

## Tier 4 — Demo packaging

| # | Item | Status |
|---|---|---|
| 40 | Main menu / title screen + seed input | todo |
| 41 | Loading screen instead of popping into a half-built world | todo |
| 42 | Quality presets (Low/Med/High/Ultra) | todo |
| 43 | Screenshot key + `screenshots/` | todo |
| 44 | Cinematic free-cam / flythrough path — for recording | todo |
| 45 | Benchmark mode — deterministic flythrough, perf regression numbers | todo |
| 46 | Packaged release — zip with DLLs + assets, icon, window title | todo |
| 47 | README with GIFs | todo |
| 48 | Crash handler / minidump | todo |

---

## How I want you to work

- One change per go-ahead unless I give a broad one. Commit locally after each, lowercase
  `TODO:` / `DONE:` / `FIX:` messages. **Never push.**
- Verify with the `mingw` preset AND run `voxelium --smoke` on **both** Release and Debug.
  Debug's container assertions catch data races Release silently tolerates.
- Profile before you optimize, and show me the numbers.
- Explain the why. For a non-obvious decision, say what the alternatives were and why this
  one wins. Comment the intent, not the mechanics.
- Small files, clear ownership. No god classes.
- Ask before adding a dependency.
- Write it so I could have written it — I need to be able to defend and modify every line.
- No copyrighted assets, ever. Textures and sounds are procedural or user-supplied.

### Invariants that must not break

- The startup mesher oracle must pass.
- The `--smoke` suite stays green (light, physics, edits, persistence round-trip + reload).
- `src/mesh/QuadFormat.hpp` stays in sync with `chunk.vert`, `shadow.vert` and `cull.comp`.
- `JobSystem` is declared after `ChunkManager` in `main` (teardown ordering).
- Never call `glGetError()` or `glFinish()` outside a debug build.
- Never block the main thread on generation, meshing, or disk I/O.
- Benchmarks swing ~20% run to run. Always measure a change back-to-back against HEAD,
  never against a number from earlier in the session.

Inspiration: <https://www.youtube.com/watch?v=CJ94gOzKqsM&t=613s>
