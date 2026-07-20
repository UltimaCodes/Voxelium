# Build Prompt: High-Performance Voxel Engine in C++

---

## Context

I'm building a voxel engine in C++ from scratch — a Minecraft-like game, but engineered properly. Minecraft's implementation is a case study in what not to do: whole-chunk remeshes on single block edits, no face merging, a naive lighting engine, single-threaded meshing for most of its life, and GC pressure from Java. I want the opposite of that. The target is a machine that renders a large voxel world at high framerate on weak hardware.

I'm a solo developer. I want to understand every design decision, not just have working code handed to me. Explain your reasoning as you go, and if I've specified something below that's wrong or suboptimal, say so and argue for the alternative before writing it.

---

## Hard constraints

- **C++20.** Use concepts, `std::span`, designated initializers, `std::jthread`. No exceptions in hot paths.
- **OpenGL 4.6 core profile with DSA** (Direct State Access). Not Vulkan — I want to ship, not spend three months before drawing a triangle. Design so a Vulkan backend is possible later, but do not abstract preemptively for it.
- **No game engine.** No Unity, Unreal, Godot, or bgfx.
- **Dependencies (ask before adding any others):**
  - GLFW — windowing/input
  - glad — GL loader (4.6 core, DSA)
  - GLM — math
  - stb_image — texture loading
  - FastNoise2 — SIMD noise
  - Dear ImGui — debug UI only
  - `ankerl::unordered_dense` — flat hash map
- **CMake** build. Must compile clean on Windows/MSVC and Linux/GCC with `-Wall -Wextra` and no warnings.
- **Zero heap allocation in the render loop.** Arenas, pools, and pre-sized buffers. If you need a scratch buffer during meshing, it's thread-local and reused.
- **Data-oriented.** No deep inheritance, no virtual dispatch anywhere near a per-voxel or per-frame path. A block is an integer ID indexing into a flat block registry, not an object.

## Performance targets

| Metric | Target |
|---|---|
| Render distance | 32 chunks minimum |
| Frame time (mid GPU, e.g. RTX 2070S) | < 4 ms at 32 chunks, shaders on |
| Frame time (integrated graphics) | < 16 ms at 12 chunks, shaders off |
| Chunk mesh time | < 1 ms per 32³ chunk, worker thread |
| Block edit → visible | < 1 frame, remesh affected chunk only |
| Per-frame CPU cost per visible chunk | ~0 (GPU-driven culling) |
| Memory | < 2 GB at 32 chunk render distance |

Build the debug HUD (frame time graph, draw calls, triangles, quads, chunk states, meshing queue depth, VRAM usage) in **milestone 2**, not at the end. I want to be able to prove every optimization.

---

## Architecture requirements

### Chunk storage

- **32³ chunks.** Power-of-two indexing so all coordinate math is shifts and masks, never `div`/`mod`.
- **Palette compression.** Each chunk stores a palette of unique block IDs plus a bit-packed index array at `ceil(log2(palette.size()))` bits per voxel. A chunk with 8 distinct block types costs 3 bits per voxel, not 16. Repack when the palette grows past a power-of-two boundary.
- **Uniform chunks allocate nothing.** All-air and all-stone chunks store a single value and no index array. Most of a world is air; it must cost near-zero.
- **World storage:** `ankerl::unordered_dense::map<ChunkPos, Chunk*>` with a proper spatial hash. Never `std::map`.
- Chunks live in a pool allocator. Unloading returns to the pool; it doesn't call the allocator.

### Meshing — binary greedy meshing

This is the core of the engine. Get it right.

1. Build **binary bitmasks** of solidity: for each of the 3 axes, each row of the chunk is a bitfield. Use `uint64_t` rows with one bit of neighbor padding on each side so face culling works correctly across chunk borders. (32 voxels + 2 padding = 34 bits in a 64-bit word. This wastes half the word — if you'd rather argue for 62³ chunks to use the full `uint64`, make the case, but note the remesh cost on edit.)
2. **Face culling with bit ops:** a face exists where a solid voxel is adjacent to a non-solid one, i.e. `faces = col & ~(col >> 1)` and `col & ~(col << 1)` for the two directions along each axis. Six passes, one per face direction.
3. **Greedy merge** the resulting binary planes into maximal quads using `std::countr_zero` / `std::countr_one` to find runs, then expand vertically while rows match. No nested per-voxel loops.
4. **Merge key must include** block ID, face direction, light level, and AO values — quads can only merge if all of those match, or you get visible seams.

### Vertex format — vertex pulling, no VBO

Do **not** emit expanded vertices into a vertex buffer. Instead:

- Pack each quad into a **single `uint64_t`** (or two `uint32_t`):
  - x, y, z: 6 bits each (0–32 inclusive) = 18 bits
  - width, height: 6 bits each = 12 bits
  - face direction: 3 bits
  - texture layer: 12 bits
  - AO, 4 corners × 2 bits = 8 bits
  - light (block + sky, 4 bits each) = 8 bits
  - Total ~61 bits. Document the exact layout in a header with named constants and static asserts.
- Store quads in an **SSBO**. Bind no vertex array attributes at all.
- Draw with `glDrawArrays(GL_TRIANGLES, 0, quadCount * 6)`. In the vertex shader, `gl_VertexID / 6` is the quad index and `gl_VertexID % 6` is the corner; unpack the quad from the SSBO and reconstruct the world position.
- Texture tiling across a merged quad: scale UVs by the quad's width/height and rely on `GL_REPEAT` per array layer.

### Ambient occlusion

- Per-vertex AO using the standard 3-neighbor test (0fps.net "Ambient occlusion for Minecraft-like worlds").
- **Flip the quad's triangulation diagonal** when the AO values are anisotropic (`a00 + a11 > a01 + a10`), or you get the classic diagonal seam artifact. This is two lines and everyone forgets it.

### Rendering pipeline

- **One persistent-mapped buffer** as a global quad arena, with a sub-allocator (free list or buddy allocator) handing out ranges to chunks. Defragment lazily when fragmentation exceeds a threshold.
- **`glMultiDrawArraysIndirect`** — one draw call for all opaque chunk geometry per frame.
- **GPU-driven culling:** a compute shader reads the chunk list, does frustum culling against per-chunk AABBs, and writes the indirect draw command buffer. Zero CPU per-chunk work per frame.
- **Cave culling (do not skip this).** For each chunk, precompute at mesh time which of the 6 faces are connected to each other through non-solid voxels (flood fill → 15 face-pair bits). At render time, flood-fill visibility outward from the camera's chunk, only traversing into a chunk through a face-pair that's actually connected and within the frustum. Underground and in dense terrain this culls the overwhelming majority of chunks. Reference: Tommo's "advanced cave culling algorithm."
- **Transparent pass** separate from opaque: back-to-front sorted, depth write off, rendered after.
- **Textures:** `GL_TEXTURE_2D_ARRAY`, never an atlas — atlases bleed at mip boundaries and force you to disable mipmapping. `NEAREST` magnification for the pixel-art look, mipmapped + anisotropic minification.

### Threading

- Job system with a thread pool of `hardware_concurrency() - 1` workers (`std::jthread`).
- Pipeline stages: **generate → light → mesh → upload.** Only upload touches the GL context, on the main thread, budgeted (e.g. max 2 ms of uploads per frame) so a burst of chunks never stalls a frame.
- Per-chunk state machine driven by `std::atomic`. The main thread never blocks on a worker.
- A block edit remeshes **only** the containing chunk, plus the 1–3 neighbors if the edit touches a chunk border. Never the column, never the region.

### Lighting

- 4 bits block light + 4 bits sky light per voxel, stored in a parallel nibble array.
- **BFS flood-fill propagation** with an explicit queue — no recursion.
- **Incremental updates on block change.** Light *removal* is the part people get wrong: it needs its own BFS pass that clears the affected region and enqueues boundary lights for re-propagation. Implement both add and remove properly.
- Sky light propagates straight down at full strength with no attenuation, spreads horizontally with attenuation.

### Terrain generation

- FastNoise2, fractal simplex with domain warping.
- 3D noise caves plus "noodle" caves (thin winding tunnels from ridged noise).
- Biomes from temperature/humidity noise mapped through splines.
- **Fully deterministic from a seed, and chunk-independent** — generating chunk (5, 3) must not require chunk (4, 3) to exist. Structures that cross chunk borders go through a deferred structure queue, not neighbor lookups.

### Shaders

Forward rendering with a depth prepass (deferred makes MSAA and the transparent water pass painful for marginal gain here). Implement in this order:

1. Cascaded shadow maps, 3–4 cascades, PCF filtering, stabilized cascades so shadows don't swim when the camera moves.
2. SSAO or GTAO layered on top of the per-vertex AO.
3. Water: normal-mapped waves, depth-based refraction, screen-space reflections.
4. Volumetric light scattering raymarched through the shadow map.
5. Bloom via the downsample/upsample chain (Jimenez, *Next Generation Post Processing in Call of Duty*), not a naive Gaussian.
6. ACES filmic tonemapping, sRGB framebuffer, exposure control.
7. *(Stretch)* TAA with jittered projection and reprojection.

Every effect gets a runtime toggle in the debug UI and a documented frame-time cost.

---

## Milestones

Each one must build, run, and be independently demoable. **Do not start the next until I confirm.**

1. **Window + GL context.** GLFW, glad, DSA, a clear color, a spinning triangle. CMake building on both platforms.
2. **Debug infrastructure.** ImGui overlay, frame time graph, GL debug callback (`glDebugMessageCallback`) wired to a logger, free-fly camera.
3. **One chunk, dumb mesh.** 32³ chunk, naive cube-per-voxel meshing, hidden faces culled. Slow and correct. Record the numbers — this is the baseline everything gets measured against.
4. **Binary greedy meshing + vertex pulling.** Replace stage 3's mesher. Show me the before/after triangle count and mesh time.
5. **World of chunks.** Chunk manager, load/unload by distance, palette-compressed storage, frustum culling on the CPU.
6. **Threading.** Job system, generation and meshing moved to workers, budgeted uploads. Flying at speed must not hitch.
7. **Indirect rendering + GPU culling.** Persistent-mapped arena, sub-allocator, `glMultiDrawArraysIndirect`, compute-shader frustum culling. Target: draw call count for the whole world = 1.
8. **Cave culling.** Connectivity graph + visibility flood fill. Measure chunks-drawn underground before and after.
9. **Lighting.** Block and sky light, flood fill, incremental add and remove on edits.
10. **Terrain + interaction.** FastNoise2 world gen, biomes, caves, raycast block placing/breaking, collision and player physics.
11. **Shaders.** The list above, in order, one at a time.
12. **Persistence.** Region file format, chunk serialization, compressed on disk.

---

## Explicitly do not

- Re-mesh a whole chunk column, region, or world on a single block edit.
- Use a texture atlas.
- Issue one draw call per chunk.
- Use `std::map`, `std::list`, or `std::shared_ptr` in any hot path.
- Put virtual functions on blocks, voxels, or anything touched per-frame.
- Call `glGetError()` or `glFinish()` outside of a debug build.
- Block the main thread on chunk generation, meshing, or disk I/O.
- Write an ECS. There is nothing to put in it yet.
- Build a "generic reusable engine." Build this specific game. Generality is earned, not assumed.
- Optimize anything without profiling it first and showing me the numbers.

---

## How I want you to work

- **One milestone at a time.** Stop and wait for me at each boundary.
- **Explain the why.** For every non-obvious decision, tell me what the alternatives were and why this one wins. If a bit-twiddling trick isn't obvious, comment the intent, not the mechanics.
- **Small files, clear ownership.** No god classes. `Chunk`, `ChunkMesher`, `ChunkManager`, `Renderer`, `LightEngine`, `TerrainGenerator`, `JobSystem` are separate units with explicit interfaces.
- **Profile before you optimize.** Show me the measurement that justifies the change.
- **Ask before adding a dependency.**
- **Write it so I could have written it.** I need to be able to defend and modify every line without you.

Start with milestone 1. Before you write any code, tell me the repo layout you're proposing and flag anything in this spec you disagree with.

https://www.youtube.com/watch?v=CJ94gOzKqsM&t=613s USE THIS AS A GOOD INSPIRATION
