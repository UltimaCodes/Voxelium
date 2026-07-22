# Post-spec pass — bug fixes, optimizations, water, trees

date: 2026-07-22
context: after all 12 milestones, a round of "eradicate bugs + optimize + add water/trees"

## bugs fixed

- **arena overflow at high render distance** (the "glitchy lag"). the quad arena was 8M
  quads; radius 16 with caves needs ~7.7M, so it ran ~92% full and called
  hardReset (glFinish, a full GPU stall) + rebuilt every few frames — chunks flickering,
  frame spikes. bumped to 16M quads / 128 MB. no more thrashing at 16.
- **light-seed vs worker-gen palette race** (caught earlier by the debug build's
  container asserts, kept fixed): the light seed only runs once a column's 5-column
  neighborhood is out of the Generating state, so it never reads a chunk mid-makeUniform.
- **per-frame O(chunks) stats scan**: the HUD's meshed/uniform/byte totals re-scanned
  every loaded chunk every frame (~5000 at radius 16). now cached and refreshed a few
  times a second.
- **cosmetic**: the "world loaded" summary logged before streaming started (0 meshed);
  now waits until work has actually happened.

## optimizations

- **shadows: 4.9 -> 3.5 ms** (radius 10). tightened cascade splits 26/80/240 -> 22/60/150.
  the shadow pass is vertex-bound (every quad drawn per cascade), so the far split is the
  real lever — 150 keeps crisp shadows out to ~5 chunks and roughly halves the far
  cascade's geometry. per-cascade frustum culling (from M11) already stopped the whole
  world being drawn into every cascade.
- **SSAO overhaul** (fixes the "blur up close"): the AO was half-res with a wide box blur
  that smeared across edges. now it carries linear view-depth in a second channel and the
  blur is bilateral (depth-weighted) — no more bleeding onto near blocks, and a tighter
  radius + contrast curve gives crisp contact AO that reads as cave/crevice smoothing
  without touching the geometry (keeps meshing maximally merged — the optimization-correct
  answer to "smooth lighting").

## sky

- real sky computed in the composite pass for pixels with no geometry (no extra draw):
  zenith->horizon gradient, warm glow around the sun, a bright sun disc, hazy below the
  horizon. replaces the flat fill colour that read as "no sky".

## water (transparent pass)

- new block: Water (non-opaque — see-through, lets light pass, no collision). the mesher
  is now two-layer: opaque geometry and water go into separate quad lists. opaque faces
  show against air OR water (you see the seabed); water faces show only against air (the
  surface + shore edges), never water-water or water-opaque.
- terrain floods air that sits over the seafloor at/below sea level (y=62) into oceans and
  lakes; caves stay dry (they're below the surface, never above it). sand beaches/seabed.
- the renderer got a second meta buffer + a transparent draw pass: frustum-culled through
  the same cull compute, drawn after opaque with alpha blending and depth-write off. water
  shader: animated wave surface, fresnel sky tint, sharp sun specular, sky-light dimming.
- cave culling now floods through water too (it's see-through).

## trees (deferred structures)

- new blocks: Wood, Leaves (opaque "fast" foliage). trees are placed deterministically:
  each chunk scans every tree-candidate cell that could reach into it (a canopy-radius of
  overlap) and places the parts that fall inside. same tree computed identically by every
  chunk it touches, so trunks/canopies cross chunk borders seamlessly with **no neighbor
  lookups** — exactly the spec's "deferred structure" requirement. trees only grow on
  grass above the beach line.

## numbers (RTX 2070S, 1280x720, radius 10)

full pipeline ~4.5 ms all-on (shadows 3.6 dominate). ~0.9 ms shaders off. arena 26 MB /
128 MB. all --smoke checks green on Release AND Debug (mesher/manager changes are
race-free). the standard surface image gained trees, water, real sky, and visible AO.

## still open (future levers)

- shadow-caster LOD / distant-chunk simplification is the remaining big FPS lever (and the
  path to the spec's 32-chunk / 4 ms target). shadows are still the dominant cost.
- terrain LOD for true long render distance.
- screen-space reflections/refraction on the water (currently stylized, no SSR).
