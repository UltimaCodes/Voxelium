#version 460 core

// depth-only pass for the shadow cascades: same quad unpack as chunk.vert, nothing else.
// keep the unpack in sync with mesh/QuadFormat.hpp.
layout(std430, binding = 0) readonly buffer Quads { uvec4 quads[]; };
layout(std430, binding = 1) readonly buffer Offsets { vec4 uOffsets[]; };

layout(location = 0) uniform mat4 uViewProj; // light view-proj, camera-relative space (0..3)
// The plane perpendicular to the light, in world units. Leaf gaps are keyed to where a ray
// crosses THIS plane, so every layer of a thick canopy makes the same decision for it.
layout(location = 4) uniform vec3 uLightX;
layout(location = 5) uniform vec3 uLightY;
layout(location = 6) uniform vec2 uCamLightXY; // camera position projected onto the same axes

// Cutout blocks (leaves) have to be alpha-tested in the shadow pass too, or a canopy full
// of holes still casts a solid slab of shadow. That needs the texture coords and the layer
// down here, which a depth-only pass would otherwise never carry.
out vec2 vUv;
out vec2 vGap;
flat out uint vLayer;

const int kTri[6] = int[6](0, 1, 2, 0, 2, 3);
const vec2 kCorner[6][4] = vec2[6][4](
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)),
    vec2[4](vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(0, 0)),
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)),
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
    vec2[4](vec2(1, 0), vec2(0, 0), vec2(0, 1), vec2(1, 1))
);
#include "texlayer.glsl"

const float kBloat = 0.001;

void main() {
    uvec4 q = quads[gl_VertexID / 6];
    int corner = kTri[gl_VertexID % 6];
    vec3 origin = vec3(uvec3(q.x & 31u, (q.x >> 5) & 31u, (q.x >> 10) & 31u));
    float w = float(((q.x >> 15) & 31u) + 1u);
    float h = float(((q.x >> 20) & 31u) + 1u);
    uint dir = (q.x >> 25) & 7u;

    // Cross-plane foliage (dir 6/7) casts no shadow: collapse it to a degenerate triangle.
    // kCorner has six entries, so indexing it at 6 would be out of bounds, and a field of
    // grass self-shadowing at cascade resolution is noise rather than detail anyway.
    if (dir >= 6u) {
        vUv = vec2(0.0);
        vGap = vec2(0.0);
        vLayer = 0u;
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
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
    vUv = uv;
    vGap = vec2(dot(world, uLightX), dot(world, uLightY)) + uCamLightXY;
    vLayer = texLayer(q.y & 255u, dir);
    gl_Position = uViewProj * vec4(world, 1.0);
}
