#version 460 core

// Vertex pulling: no vertex attributes at all. gl_VertexID / 6 picks the quad out of
// the SSBO, gl_VertexID % 6 picks the corner. Bit layout must match
// src/mesh/QuadFormat.hpp — that header is the source of truth.
layout(std430, binding = 0) readonly buffer Quads { uvec2 quads[]; };
layout(std430, binding = 1) readonly buffer Offsets { vec4 uOffsets[]; };

layout(location = 0) uniform mat4 uViewProj; // camera-relative: view is at the origin

out vec3 vWorldRel;
flat out vec3 vAlbedo;
flat out uint vDir;
flat out vec2 vLight; // x = block light, y = sky light, 0..1

const int kTri[6] = int[6](0, 1, 2, 0, 2, 3);
const vec2 kCorner[6][4] = vec2[6][4](
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)), // +X
    vec2[4](vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(0, 0)), // -X
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)), // +Y
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)), // -Y
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)), // +Z
    vec2[4](vec2(1, 0), vec2(0, 0), vec2(0, 1), vec2(1, 1))  // -Z
);

// placeholder until the texture array lands
const vec3 kBlockColor[9] = vec3[9](
    vec3(0.0),
    vec3(0.490, 0.490, 0.490), // stone
    vec3(0.525, 0.376, 0.263), // dirt
    vec3(0.416, 0.667, 0.251), // grass
    vec3(1.000, 0.850, 0.550), // glow
    vec3(0.180, 0.360, 0.620), // water (opaque path never uses this)
    vec3(0.420, 0.300, 0.180), // wood
    vec3(0.240, 0.470, 0.190), // leaves
    vec3(0.855, 0.800, 0.590)  // sand
);

// greedy T-junction pinhole fix — see docs/M4.md
const float kBloat = 0.001;

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

    vWorldRel = pos + uOffsets[gl_DrawID].xyz;
    vAlbedo = kBlockColor[min(block, 8u)];
    vDir = dir;
    vLight = vec2(float((q.y >> 20) & 15u), float((q.y >> 24) & 15u)) / 15.0;
    gl_Position = uViewProj * vec4(vWorldRel, 1.0);
}
