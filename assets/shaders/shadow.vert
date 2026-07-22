#version 460 core

// depth-only pass for the shadow cascades: same quad unpack as chunk.vert, nothing else.
// keep the unpack in sync with mesh/QuadFormat.hpp.
layout(std430, binding = 0) readonly buffer Quads { uvec2 quads[]; };
layout(std430, binding = 1) readonly buffer Offsets { vec4 uOffsets[]; };

layout(location = 0) uniform mat4 uViewProj; // light view-proj, camera-relative space

const int kTri[6] = int[6](0, 1, 2, 0, 2, 3);
const vec2 kCorner[6][4] = vec2[6][4](
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)),
    vec2[4](vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(0, 0)),
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)),
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
    vec2[4](vec2(1, 0), vec2(0, 0), vec2(0, 1), vec2(1, 1))
);
const float kBloat = 0.001;

void main() {
    uvec2 q = quads[gl_VertexID / 6];
    int corner = kTri[gl_VertexID % 6];
    vec3 origin = vec3(uvec3(q.x & 31u, (q.x >> 5) & 31u, (q.x >> 10) & 31u));
    float w = float(((q.x >> 15) & 31u) + 1u);
    float h = float(((q.x >> 20) & 31u) + 1u);
    uint dir = (q.x >> 25) & 7u;

    vec2 c = kCorner[dir][corner];
    vec2 uv = c * vec2(w, h) + (c * 2.0 - 1.0) * kBloat;
    vec3 pos = origin;
    if (dir == 0u)      pos += vec3(1.0, uv.y, uv.x);
    else if (dir == 1u) pos += vec3(0.0, uv.y, uv.x);
    else if (dir == 2u) pos += vec3(uv.x, 1.0, uv.y);
    else if (dir == 3u) pos += vec3(uv.x, 0.0, uv.y);
    else if (dir == 4u) pos += vec3(uv.x, uv.y, 1.0);
    else                pos += vec3(uv.x, uv.y, 0.0);

    gl_Position = uViewProj * vec4(pos + uOffsets[gl_DrawID].xyz, 1.0);
}
