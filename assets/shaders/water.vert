#version 460 core

// transparent water surface: same packed-quad unpack as chunk.vert, but the top faces
// ripple and sit slightly below the block top so the surface reads as liquid.
layout(std430, binding = 0) readonly buffer Quads { uvec2 quads[]; };
layout(std430, binding = 1) readonly buffer Offsets { vec4 uOffsets[]; };

layout(location = 0) uniform mat4 uViewProj; // occupies locations 0..3
layout(location = 4) uniform float uTime;

out vec3 vWorldRel;
flat out uint vDir;
out float vSky; // interpolated: water gets the same per-corner shade the opaque pass does

const int kTri[6] = int[6](0, 1, 2, 0, 2, 3);
const vec2 kCorner[6][4] = vec2[6][4](
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)),
    vec2[4](vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(0, 0)),
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)),
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
    vec2[4](vec2(1, 0), vec2(0, 0), vec2(0, 1), vec2(1, 1))
);

void main() {
    uvec2 q = quads[gl_VertexID / 6];
    int corner = kTri[gl_VertexID % 6];
    vec3 origin = vec3(uvec3(q.x & 31u, (q.x >> 5) & 31u, (q.x >> 10) & 31u));
    float w = float(((q.x >> 15) & 31u) + 1u);
    float h = float(((q.x >> 20) & 31u) + 1u);
    uint dir = (q.x >> 25) & 7u;

    vec2 uv = kCorner[dir][corner] * vec2(w, h);
    vec3 pos = origin;
    if (dir == 0u)      pos += vec3(1.0, uv.y, uv.x);
    else if (dir == 1u) pos += vec3(0.0, uv.y, uv.x);
    else if (dir == 2u) pos += vec3(uv.x, 1.0, uv.y);
    else if (dir == 3u) pos += vec3(uv.x, 0.0, uv.y);
    else if (dir == 4u) pos += vec3(uv.x, uv.y, 1.0);
    else                pos += vec3(uv.x, uv.y, 0.0);

    vec3 world = pos + uOffsets[gl_DrawID].xyz;
    // Flowing water carries its depth in the block id (Water = 8 = full source, 14..20 are
    // L7..L1). A shallower cell sits lower in its voxel, so a spreading sheet visibly
    // thins out toward its edge instead of every cell being a full block of water.
    // Per-corner surface level, packed 4 x 3 bits into the r/g/b nibbles of the light word
    // by GreedyMesher::waterCorners — water only ever reads sky out of that word, so those
    // twelve bits were free. Each corner is the average depth of the columns meeting there,
    // which is what makes a thinning sheet RAMP; using the cell's own level gave every
    // depth change a hard step, and that was the jagged water.
    uint corners = q.y >> 16;
    float level = float(((corners >> (uint(corner) * 3u)) & 7u) + 1u);
    // How far this cell's surface sits below the top of its voxel.
    float drop = 0.10 + (8.0 - level) * 0.105;

    // THE SWELL IS NOT DISPLACED ANY MORE, and that is a fix rather than a simplification.
    // A vertical wave sampled per VERTEX is interpolated linearly across the face, and water
    // quads are greedy-merged up to 32x32 — so a large quad and a small neighbour disagree
    // along their shared edge by however much the wave curves between the large quad's
    // corners. That gap opens and closes as the swell moves: the visible separation between
    // water blocks. Any fix that keeps displacement has to defeat the merge (cap quad size,
    // or tessellate), and both cost geometry for motion the eye reads mostly off the
    // highlights anyway.
    //
    // So the surface stays flat at its depth-derived level, which is per-corner and shared,
    // and therefore watertight by construction. The wave lives entirely in the fragment
    // normal (waveGrad in water.frag), which is also where the good Minecraft packs put it.

    // The drop used to be applied to the TOP FACE ONLY. Side faces then still spanned the
    // full block, so every shallow cell had walls standing proud of its own surface: a rim
    // of water sticking up around a lower top, with a gap you could see straight through
    // where the neighbour was shallower still. That is the jagged, glitchy shoreline.
    //
    // The upper edge of a side face is the same surface, so it has to move with it, and
    // both read the same per-corner level — so the wall meets the top exactly.
    bool topEdge = (dir != 2u && dir != 3u) && uv.y >= h - 0.001;
    if (dir == 2u || topEdge)
        world.y -= drop;

    vWorldRel = world;
    vDir = dir;
    // sky is the top nibble of the light word, scaled by this corner's shade (4 corners x
    // 3 bits, low byte in hi 8..15 + high nibble in lo 28..31 — see mesh/QuadFormat.hpp).
    // Water never merged across differing light either, so this costs it nothing.
    uint shade12 = ((q.y >> 8u) & 255u) | (((q.x >> 28u) & 15u) << 8u);
    float shade = float((shade12 >> (uint(corner) * 3u)) & 7u) / 7.0;
    vSky = (float((q.y >> 28) & 15u) / 15.0) * shade;
    gl_Position = uViewProj * vec4(world, 1.0);
}
