#version 460 core

// One camera-facing quad per particle, vertex-pulled: gl_VertexID / 6 picks the particle,
// gl_VertexID % 6 picks the corner. Same no-VBO approach as the chunk mesh.
struct P {
    vec4 posSize;  // camera-relative centre, then half-extent
    vec4 tint;     // rgb tint, then alpha
    vec4 layerPad; // .x = block texture array layer, < 0 for a flat colour
};
layout(std430, binding = 0) readonly buffer Parts { P parts[]; };

layout(location = 0) uniform mat4 uViewProj; // 0..3
layout(location = 4) uniform vec3 uRight;    // camera basis, so quads always face the eye
layout(location = 5) uniform vec3 uUp;

out vec2 vUv;
out vec4 vTint;
flat out float vLayer;

const vec2 kQuad[6] =
    vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    P p = parts[gl_VertexID / 6];
    vec2 c = kQuad[gl_VertexID % 6];
    vec3 world = p.posSize.xyz + (uRight * c.x + uUp * c.y) * p.posSize.w;
    vUv = c * 0.5 + 0.5;
    vTint = p.tint;
    vLayer = p.layerPad.x;
    gl_Position = uViewProj * vec4(world, 1.0);
}
