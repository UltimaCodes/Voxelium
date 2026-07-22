#version 460 core

// Vertex pulling: no vertex attributes at all. gl_VertexID / 6 picks the quad out of
// the SSBO, gl_VertexID % 6 picks the corner. Bit layout must match
// src/mesh/QuadFormat.hpp — that header is the source of truth.
layout(std430, binding = 0) readonly buffer Quads { uvec2 quads[]; };
layout(std430, binding = 1) readonly buffer Offsets { vec4 uOffsets[]; };

layout(location = 0) uniform mat4 uViewProj; // camera-relative: view is at the origin
layout(location = 20) uniform float uTime;   // for leaf sway

out vec3 vWorldRel;
out vec2 vUv;         // tiling texture coords (0..w, 0..h)
flat out uint vLayer; // texture-array layer
flat out uint vDir;
flat out vec2 vLight; // block, sky (0..1)

const int kTri[6] = int[6](0, 1, 2, 0, 2, 3);
const vec2 kCorner[6][4] = vec2[6][4](
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)), // +X
    vec2[4](vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(0, 0)), // -X
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)), // +Y
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)), // -Y
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)), // +Z
    vec2[4](vec2(1, 0), vec2(0, 0), vec2(0, 1), vec2(1, 1))  // -Z
);

// block id + face dir -> texture-array layer. mirrors BlockTextures layer order.
uint texLayer(uint block, uint dir) {
    if (block == 1u) return 0u;                              // stone
    if (block == 2u) return 1u;                              // dirt
    if (block == 3u) return dir == 2u ? 2u : (dir == 3u ? 1u : 3u); // grass top/bottom/side
    if (block == 4u) return 4u;                              // glow
    if (block == 6u) return (dir == 2u || dir == 3u) ? 6u : 5u;     // wood top/side
    if (block == 7u) return 7u;                              // leaves
    if (block == 8u) return 8u;                              // sand
    return 0u;
}

const float kBloat = 0.001; // greedy T-junction pinhole fix (docs/M4)

void main() {
    uvec2 q = quads[gl_VertexID / 6];
    int corner = kTri[gl_VertexID % 6];

    vec3 origin = vec3(uvec3(q.x & 31u, (q.x >> 5) & 31u, (q.x >> 10) & 31u));
    float w = float(((q.x >> 15) & 31u) + 1u);
    float h = float(((q.x >> 20) & 31u) + 1u);
    uint dir = (q.x >> 25) & 7u;
    uint block = q.y & 4095u;

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
    if (block == 7u) {
        float ph = world.x * 0.5 + world.z * 0.5;
        world.x += 0.06 * sin(uTime * 1.6 + ph);
        world.z += 0.05 * sin(uTime * 1.9 + ph * 1.3);
        world.y += 0.03 * sin(uTime * 2.2 + ph * 0.7);
    }

    vWorldRel = world;
    vUv = c * vec2(w, h); // tile once per block, GL_REPEAT wraps it
    vLayer = texLayer(block, dir);
    vDir = dir;
    vLight = vec2(float((q.y >> 20) & 15u), float((q.y >> 24) & 15u)) / 15.0;
    gl_Position = uViewProj * vec4(world, 1.0);
}
