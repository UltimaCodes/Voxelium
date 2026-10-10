#version 460 core

// How finely each water quad is cut up for the swell to move it.
//
// The swell used to be applied per vertex of the meshed quads and was taken out again, because
// water quads are greedy-merged up to 32x32: a big quad's edge is one straight line where its
// small neighbours have corners along it, the wave curves between those corners, and the gap
// opened and closed as the swell moved. Cutting every live edge at every half block (1 a block
// on the two lowest presets) puts the vertices of BOTH sides of any shared edge on the same
// lattice, and the displacement is a function of position alone, so they meet exactly.
//
// An edge stays whole when nothing along it can move: the mesher gave both its ends no wave
// strength (and along a merged edge every lattice point has its ends' strength), or all of it is
// past the fade. Whatever sits on the other side of such an edge is still for the same reason.
layout(vertices = 4) out;

in vec3 vcLocal[];
in vec3 vcOffset[];
in float vcSky[];
in float vcWave[];
in uint vcDir[];

out vec3 tcLocal[];
out vec3 tcOffset[];
out float tcSky[];
out float tcWave[];
out uint tcDir[];

layout(location = 31) uniform float uWaveScale;
layout(location = 32) uniform vec2 uWaveFade;
layout(location = 35) uniform float uTessDensity; // vertices a block along a live edge

float edgeLevel(int a, int b) {
    if (vcDir[0] != 2u || uWaveScale <= 0.0 || (vcWave[a] <= 0.0 && vcWave[b] <= 0.0))
        return 1.0;
    vec3 pa = vcLocal[a] + vcOffset[a];
    vec3 ab = (vcLocal[b] + vcOffset[b]) - pa;
    float t = clamp(dot(-pa, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    if (length(pa + ab * t) >= uWaveFade.y + 1.0)
        return 1.0;
    return clamp(round(length(ab.xz)) * uTessDensity, 1.0, 64.0);
}

void main() {
    tcLocal[gl_InvocationID] = vcLocal[gl_InvocationID];
    tcOffset[gl_InvocationID] = vcOffset[gl_InvocationID];
    tcSky[gl_InvocationID] = vcSky[gl_InvocationID];
    tcWave[gl_InvocationID] = vcWave[gl_InvocationID];
    tcDir[gl_InvocationID] = vcDir[gl_InvocationID];
    if (gl_InvocationID == 0) {
        // corners 0..3 sit at domain (0,0) (1,0) (1,1) (0,1); outer 0 is the u = 0 edge (3-0),
        // 1 is v = 0 (0-1), 2 is u = 1 (1-2), 3 is v = 1 (2-3)
        float e0 = edgeLevel(3, 0), e1 = edgeLevel(0, 1), e2 = edgeLevel(1, 2), e3 = edgeLevel(2, 3);
        gl_TessLevelOuter[0] = e0;
        gl_TessLevelOuter[1] = e1;
        gl_TessLevelOuter[2] = e2;
        gl_TessLevelOuter[3] = e3;
        gl_TessLevelInner[0] = max(e1, e3);
        gl_TessLevelInner[1] = max(e0, e2);
    }
}
