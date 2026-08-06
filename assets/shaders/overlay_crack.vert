#version 460 core

// The six faces of one voxel, vertex-pulled from gl_VertexID (no buffers), drawn over the
// block the player is breaking. Corners are camera-relative to pair with the same
// proj * viewAtOrigin the world uses.
layout(location = 0) uniform mat4 uViewProj; // 0..3
layout(location = 4) uniform vec3 uMin;      // camera-relative min corner of the voxel

out vec2 vUv;

// per face: origin then the two edge vectors, in the order +X -X +Y -Y +Z -Z
const vec3 kOrigin[6] = vec3[6](vec3(1, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0), vec3(0, 0, 1),
                                vec3(1, 0, 1), vec3(0, 0, 0));
const vec3 kU[6] = vec3[6](vec3(0, 0, 1), vec3(0, 0, -1), vec3(1, 0, 0), vec3(-1, 0, 0),
                           vec3(-1, 0, 0), vec3(1, 0, 0));
const vec3 kV[6] = vec3[6](vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 0, 1),
                           vec3(0, 1, 0), vec3(0, 1, 0));
const vec2 kQuad[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1),
                              vec2(0, 1));

void main() {
    int face = gl_VertexID / 6;
    vec2 c = kQuad[gl_VertexID % 6];
    vec3 p = kOrigin[face] + kU[face] * c.x + kV[face] * c.y;
    // lift off the block's own surface so the crack wins the depth test instead of
    // z-fighting the face it is drawn on
    p += (p - 0.5) * 0.006;
    vUv = c;
    gl_Position = uViewProj * vec4(uMin + p, 1.0);
}
