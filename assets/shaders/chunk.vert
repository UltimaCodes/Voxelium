#version 460 core

// Vertex pulling: no vertex attributes at all. gl_VertexID / 6 picks the quad out of
// the SSBO, gl_VertexID % 6 picks the corner. Bit layout must match
// src/mesh/QuadFormat.hpp — that header is the source of truth.
layout(std430, binding = 0) readonly buffer Quads { uvec4 quads[]; };
layout(std430, binding = 1) readonly buffer Offsets { vec4 uOffsets[]; };

layout(location = 0) uniform mat4 uViewProj; // camera-relative: view is at the origin
layout(location = 20) uniform float uTime;   // for leaf sway
layout(location = 24) uniform vec3 uCamPos;  // camera world position, for wind gusts

out vec3 vWorldRel;
out vec2 vUv;         // tiling texture coords (0..w, 0..h)
flat out uint vLayer; // texture-array layer
flat out uint vDir;
flat out vec4 vLight; // block light r,g,b then sky (0..1), the quad's brightest corner
out float vShade;     // per-corner AO x light, interpolated -> smooth corner shading

const int kTri[6] = int[6](0, 1, 2, 0, 2, 3);
const vec2 kCorner[6][4] = vec2[6][4](
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)), // +X
    vec2[4](vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(0, 0)), // -X
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)), // +Y
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)), // -Y
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)), // +Z
    vec2[4](vec2(1, 0), vec2(0, 0), vec2(0, 1), vec2(1, 1))  // -Z
);

#include "texlayer.glsl"


// corner order for the cross planes, matching kTri
const vec2 kQuadCorner[4] = vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0));

const float kBloat = 0.001; // greedy T-junction pinhole fix (docs/M4)

// Wind comes in gusts that sweep across the ground rather than every blade moving in step:
// a slow travelling wave in WORLD space (so walking does not drag it along), squared so most of
// the time it is a breeze and now and then it leans everything over.
float windGust(vec3 camRelPos) {
    vec3 w = camRelPos + uCamPos;
    float g = 0.5 + 0.5 * sin(dot(w.xz, vec2(0.047, 0.031)) + uTime * 0.45);
    return 0.55 + 1.2 * g * g;
}

void main() {
    uvec4 q = quads[gl_VertexID / 6];
    int corner = kTri[gl_VertexID % 6];

    vec3 origin = vec3(uvec3(q.x & 31u, (q.x >> 5) & 31u, (q.x >> 10) & 31u));
    float w = float(((q.x >> 15) & 31u) + 1u);
    float h = float(((q.x >> 20) & 31u) + 1u);
    uint dir = (q.x >> 25) & 7u;
    uint block = q.y & 255u;

    // dir 6/7 are the two diagonal planes of a cross-plane foliage voxel — the values the
    // 3-bit dir field never used. They are quads like any other, just not axis aligned.
    if (dir >= 6u) {
        vec2 cc = kQuadCorner[corner];
        vec3 p = (dir == 6u) ? vec3(cc.x, cc.y, cc.x) : vec3(cc.x, cc.y, 1.0 - cc.x);
        vec3 wpos = origin + p + uOffsets[gl_DrawID].xyz;
        // Backface culling would eat whichever plane faces away, and there is no second
        // winding stored. Decide the winding here instead: flip the corner order when the
        // quad's centre is on the far side of its own plane from the eye (the eye is the
        // origin in camera-relative space), which gives a double-sided quad for free.
        vec3 n = (dir == 6u) ? vec3(0.7071, 0.0, -0.7071) : vec3(0.7071, 0.0, 0.7071);
        vec3 centre = origin + vec3(0.5) + uOffsets[gl_DrawID].xyz;
        if (dot(n, centre) > 0.0) {
            cc = kQuadCorner[3 - corner];
            p = (dir == 6u) ? vec3(cc.x, cc.y, cc.x) : vec3(cc.x, cc.y, 1.0 - cc.x);
            wpos = origin + p + uOffsets[gl_DrawID].xyz;
        }
        float ph = wpos.x * 0.5 + wpos.z * 0.5;
        if (block == 56u || block == 57u || (block >= 63u && block <= 65u)) {
            // Underwater plants sway with the current, and it has to be a function of world
            // height rather than of this block's own blade: kelp is stacked blocks, and a root
            // that stays put under a tip that moves would kink at every join.
            float sw = sin(uTime * 0.9 + wpos.y * 0.55 + ph) * 0.11 * (block == 57u ? 1.0 : 0.6);
            wpos.x += sw;
            wpos.z += sw * 0.6 * cos(uTime * 0.7 + wpos.y * 0.4);
        } else {
            // same wind wobble the leaves get, scaled by height up the blade so the root stays
            float gust = windGust(wpos);
            wpos.x += 0.09 * cc.y * sin(uTime * 1.7 + ph) * gust;
            wpos.z += 0.07 * cc.y * sin(uTime * 2.0 + ph * 1.3) * gust;
        }
        vWorldRel = wpos;
        // Flip V: cc.y is 0 at the BOTTOM of the voxel, but texture row 0 is the TOP of
        // the tile. Without this the blades render upside down, which also hangs them from
        // the top of the block — the "floating grass".
        vUv = vec2(cc.x, 1.0 - cc.y);
        vLayer = texLayer(block, dir);
        vDir = dir;
        uint sh12 = (q.y >> 8u) & 0xFFFu;
        vShade = float((sh12 >> (uint(corner) * 3u)) & 7u) / 7.0;
        vLight = vec4(float(q.z & 255u), float((q.z >> 8) & 255u), float((q.z >> 16) & 255u),
                      float((q.z >> 24) & 255u)) / 255.0;
        gl_Position = uViewProj * vec4(wpos, 1.0);
        return;
    }

    vec2 c = kCorner[dir][corner];
    vec2 uv = c * vec2(w, h) + (c * 2.0 - 1.0) * kBloat;
    vec3 pos = origin;
    if (dir == 0u)      pos += vec3(1.0, uv.y, uv.x);
    else if (dir == 1u) pos += vec3(0.0, uv.y, uv.x);
    else if (dir == 2u) pos += vec3(uv.x, 1.0, uv.y);
    else if (dir == 3u) pos += vec3(uv.x, 0.0, uv.y);
    else if (dir == 4u) pos += vec3(uv.x, uv.y, 1.0);
    else                pos += vec3(uv.x, uv.y, 0.0);

    vec3 world = pos + uOffsets[gl_DrawID].xyz;

    // leaf sway: gentle wind wobble driven by world position + time
    if (block == 7u || block == 54u || block == 70u || block == 71u) {
        float ph = world.x * 0.5 + world.z * 0.5;
        float gust = windGust(world);
        world.x += 0.06 * sin(uTime * 1.6 + ph) * gust;
        world.z += 0.05 * sin(uTime * 1.9 + ph * 1.3) * gust;
        world.y += 0.03 * sin(uTime * 2.2 + ph * 0.7) * gust;
    }

    // Per-corner vertex brightness: 4 corners x 3 bits, 7 = full. Lives entirely inside w1
    // (bits 8..19) — mesh/QuadFormat.hpp asserts the exact shift.
    uint shade12 = (q.y >> 8u) & 0xFFFu;
    vShade = float((shade12 >> (uint(corner) * 3u)) & 7u) / 7.0;

    vWorldRel = world;
    vUv = c * vec2(w, h); // tile once per block, GL_REPEAT wraps it
    vLayer = texLayer(block, dir);
    vDir = dir;
    // w2 is the packed light word verbatim: r, g, b, sky bytes in that order
    vLight = vec4(float(q.z & 255u), float((q.z >> 8) & 255u), float((q.z >> 16) & 255u),
                  float((q.z >> 24) & 255u)) / 255.0;
    gl_Position = uViewProj * vec4(world, 1.0);
}
