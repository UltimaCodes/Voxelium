#version 460 core

// transparent water surface: same packed-quad unpack as chunk.vert, but the top faces
// ripple and sit slightly below the block top so the surface reads as liquid.
layout(std430, binding = 0) readonly buffer Quads { uvec2 quads[]; };
layout(std430, binding = 1) readonly buffer Offsets { vec4 uOffsets[]; };

layout(location = 0) uniform mat4 uViewProj; // occupies locations 0..3
layout(location = 4) uniform float uTime;

out vec3 vWorldRel;
flat out uint vDir;
flat out float vSky;

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
    if (dir == 2u) { // top surface: drop under the block top, organic bob
        vec2 p = world.xz;
        float wave = 0.055 * sin(dot(vec2(0.80, 0.20), p) * 0.28 + uTime * 1.1) +
                     0.040 * sin(dot(vec2(-0.30, 0.95), p) * 0.41 + uTime * 1.5) +
                     0.024 * sin(dot(vec2(0.55, -0.72), p) * 0.67 + uTime * 2.1);
        world.y -= 0.10 - wave;
    }

    vWorldRel = world;
    vDir = dir;
    vSky = float((q.y >> 28) & 15u) / 15.0; // sky is the top nibble of the light word
    gl_Position = uViewProj * vec4(world, 1.0);
}
