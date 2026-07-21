#version 460 core

// Vertex pulling: no vertex attributes at all. gl_VertexID / 6 picks the quad out of
// the SSBO, gl_VertexID % 6 picks the corner. Bit layout must match
// src/mesh/QuadFormat.hpp — that header is the source of truth.
layout(std430, binding = 0) readonly buffer Quads { uvec2 quads[]; };

layout(location = 0) uniform mat4 uViewProj;   // camera-relative: view is at the origin
layout(location = 4) uniform vec3 uChunkOffset; // chunk origin minus camera position

out vec3 vColor;

// two triangles per quad: corners (0,1,2) and (0,2,3)
const int kTri[6] = int[6](0, 1, 2, 0, 2, 3);

// corner order per dir in plane (u,v) space, CCW seen from outside — same winding the
// naive mesher used, so backface culling behaves identically
const vec2 kCorner[6][4] = vec2[6][4](
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)), // +X
    vec2[4](vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(0, 0)), // -X
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)), // +Y
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)), // -Y
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)), // +Z
    vec2[4](vec2(1, 0), vec2(0, 0), vec2(0, 1), vec2(1, 1))  // -Z
);

const float kShade[6] = float[6](0.80, 0.80, 1.00, 0.50, 0.65, 0.65);

// placeholder until the texture array lands; matches the M3 vertex colors
const vec3 kBlockColor[4] = vec3[4](
    vec3(0.0),                    // air, never drawn
    vec3(0.490, 0.490, 0.490),    // stone
    vec3(0.525, 0.376, 0.263),    // dirt
    vec3(0.416, 0.667, 0.251)     // grass
);

// Greedy meshing creates T-junctions (a quad corner in the middle of a neighbor's
// edge); interpolation along the long edge doesn't exactly hit the transformed corner,
// so single-pixel cracks open. Bloating every quad a hair in-plane makes neighbors
// overlap by a sub-pixel strip — coplanar overlap is invisible, holes are not.
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

    gl_Position = uViewProj * vec4(pos + uChunkOffset, 1.0);
    vColor = kBlockColor[min(block, 3u)] * kShade[dir];
}
