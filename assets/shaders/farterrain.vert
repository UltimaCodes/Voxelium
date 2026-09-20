#version 460 core

// The far heightfield: a grid of columns, vertex-pulled from a buffer of (height, colour,
// normal). Everything is worked out in camera-relative space like the chunks are, so the world's
// distance from the origin costs nothing.
layout(std430, binding = 0) readonly buffer Verts { uint data[]; };

layout(location = 0) uniform mat4 uViewProj;
layout(location = 4) uniform vec3 uOriginRel; // grid corner relative to the camera (xz)
layout(location = 5) uniform float uCamY;
layout(location = 6) uniform float uInner;    // the streamed world reaches this far
layout(location = 10) uniform float uSpacing;
layout(location = 11) uniform int uCells;
layout(location = 12) uniform vec2 uOriginAbs; // grid corner in world x/z, for surface detail

out vec3 vColor;
out vec3 vNormal;
out vec3 vRel;
out vec2 vWorld;
flat out float vWater;

const ivec2 kCorner[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 0),
                                  ivec2(1, 1), ivec2(0, 1));

void main() {
    int cell = gl_VertexID / 6;
    ivec2 ij = ivec2(cell % uCells, cell / uCells) + kCorner[gl_VertexID % 6];
    int idx = (ij.y * (uCells + 1) + ij.x) * 3;
    float h = uintBitsToFloat(data[idx]);
    uint rgbw = data[idx + 1];
    uint nrm = data[idx + 2];

    vec3 rel = vec3(uOriginRel.x + float(ij.x) * uSpacing, h - uCamY, uOriginRel.z + float(ij.y) * uSpacing);
    // Under the real terrain wherever the real terrain is: dip a few blocks approaching the seam
    // so the streamed chunks always win the depth test and this never pokes through them.
    rel.y -= 5.0 * (1.0 - smoothstep(uInner, uInner + 220.0, length(rel.xz)));

    vColor = vec3(float(rgbw & 255u), float((rgbw >> 8) & 255u), float((rgbw >> 16) & 255u)) / 255.0;
    vWater = float((rgbw >> 24) & 255u);
    vNormal = vec3(float(int(nrm << 24) >> 24), float(int(nrm << 16) >> 24), float(int(nrm << 8) >> 24)) / 127.0;
    vRel = rel;
    vWorld = uOriginAbs + vec2(ij) * uSpacing;
    gl_Position = uViewProj * vec4(rel, 1.0);
}
