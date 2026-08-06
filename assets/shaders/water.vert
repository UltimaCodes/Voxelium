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
    uint blockId = q.y & 255u;
    float level = blockId == 5u ? 8.0 : (blockId >= 14u && blockId <= 20u
                                             ? float(1u + (20u - blockId))
                                             : 8.0);
    // How far this cell's surface sits below the top of its voxel, plus the swell.
    float drop = 0.10 + (8.0 - level) * 0.105;
    vec2 p = world.xz;
    float wave = 0.055 * sin(dot(vec2(0.80, 0.20), p) * 0.28 + uTime * 1.1) +
                 0.040 * sin(dot(vec2(-0.30, 0.95), p) * 0.41 + uTime * 1.5) +
                 0.024 * sin(dot(vec2(0.55, -0.72), p) * 0.67 + uTime * 2.1);

    // The drop used to be applied to the TOP FACE ONLY. Side faces then still spanned the
    // full block, so every shallow cell had walls standing proud of its own surface: a rim
    // of water sticking up around a lower top, with a gap you could see straight through
    // where the neighbour was shallower still. That is the jagged, glitchy shoreline.
    //
    // The upper edge of a side face is the same surface, so it has to move with it — and
    // the wave is sampled from that corner's own world position, so the wall meets the top
    // exactly instead of shearing away from it as the swell moves.
    bool topEdge = (dir != 2u && dir != 3u) && uv.y >= h - 0.001;
    if (dir == 2u || topEdge)
        world.y -= drop - wave;

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
