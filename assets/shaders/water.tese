#version 460 core

// Places each vertex of the subdivided water surface and lifts it by the swell. The domain's
// corners (0,0) (1,0) (1,1) (0,1) are the quad's corners 0..3 in the order they go round the face
// (water.vert), so counter-clockwise in the domain is the face's own front side.
layout(quads, equal_spacing, ccw) in;

in vec3 tcLocal[];
in vec3 tcOffset[];
in float tcSky[];
in float tcWave[];
in uint tcDir[];

out vec3 vWorldRel; // camera-relative world position
flat out uint vDir;
out float vSky;
out float vStrength; // 0..1 the mesher's wave strength here: shoaling, shores, overhangs

layout(location = 0) uniform mat4 uViewProj;
layout(location = 35) uniform float uTessDensity;

#include "waves.glsl"

void main() {
    float u = gl_TessCoord.x, v = gl_TessCoord.y;
    vec4 wt = vec4((1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v);
    vec3 local = tcLocal[0] * wt.x + tcLocal[1] * wt.y + tcLocal[2] * wt.z + tcLocal[3] * wt.w;
    float sky = dot(vec4(tcSky[0], tcSky[1], tcSky[2], tcSky[3]), wt);
    float wave = dot(vec4(tcWave[0], tcWave[1], tcWave[2], tcWave[3]), wt);
    uint dir = tcDir[0];

    if (dir == 2u && uWaveScale > 0.0) {
        // Onto the lattice exactly. The tessellator's coordinates are fixed point, so without this
        // a vertex on a long edge lands a hair off the corner of the small quad beside it.
        local.xz = round(local.xz * uTessDensity) / uTessDensity;
    }
    vec3 rel = local + tcOffset[0];
    if (dir == 2u && uWaveScale > 0.0)
        rel.y += swell(rel.xz).x * wave * uWaveScale * waveFade(length(rel));
    vWorldRel = rel;
    vDir = dir;
    vSky = sky;
    vStrength = wave;
    gl_Position = uViewProj * vec4(rel, 1.0);
}
