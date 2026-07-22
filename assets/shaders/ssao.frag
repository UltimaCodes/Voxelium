#version 460 core

// half-res SSAO from the depth buffer alone: view position reconstructed from the
// infinite reversed-z depth, flat normals from derivatives (voxel faces are flat, so
// dFdx/dFdy are exact away from edges). output is AO in .r and linear view depth in .g
// so the bilateral blur + composite can reject samples across depth edges (no smear).

in vec2 vUv;
layout(location = 0) out vec2 outAoDepth;

layout(binding = 0) uniform sampler2D uDepth;
layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;

const vec3 kKernel[16] = vec3[16](
    vec3( 0.53,  0.10,  0.30), vec3(-0.28,  0.36,  0.21), vec3( 0.10, -0.58,  0.45),
    vec3(-0.62, -0.15,  0.37), vec3( 0.21,  0.44,  0.63), vec3(-0.11, -0.20,  0.19),
    vec3( 0.35, -0.29,  0.71), vec3(-0.44,  0.52,  0.48), vec3( 0.66,  0.31,  0.55),
    vec3(-0.19,  0.08,  0.09), vec3( 0.05,  0.64,  0.30), vec3(-0.35, -0.51,  0.60),
    vec3( 0.48, -0.44,  0.24), vec3(-0.57,  0.22,  0.66), vec3( 0.14,  0.71,  0.12),
    vec3(-0.08, -0.36,  0.42));

float rawDepthToViewZ(float d) { return -uNear / max(d, 1e-7); } // negative, toward -z

vec3 viewPos(vec2 uv) {
    float z = rawDepthToViewZ(texture(uDepth, uv).r);
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * uProjScale.x * -z, ndc.y * uProjScale.y * -z, z);
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec3 P = viewPos(vUv);
    if (P.z < -400.0) { // sky
        outAoDepth = vec2(1.0, P.z);
        return;
    }
    vec3 N = normalize(cross(dFdx(P), dFdy(P)));

    float a = hash(gl_FragCoord.xy) * 6.2831853;
    vec3 rot = vec3(cos(a), sin(a), 0.0);
    vec3 T = normalize(rot - N * dot(rot, N));
    vec3 B = cross(N, T);

    const float kRadius = 0.55; // tighter than before: contact AO, not a soft halo
    float occlusion = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec3 s = P + (T * kKernel[i].x + B * kKernel[i].y + N * kKernel[i].z) * kRadius;
        vec2 suv = vec2(s.x / (uProjScale.x * -s.z), s.y / (uProjScale.y * -s.z)) * 0.5 + 0.5;
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0)
            continue;
        float sceneZ = rawDepthToViewZ(texture(uDepth, suv).r);
        if (sceneZ > s.z + 0.02) {
            float range = smoothstep(0.0, 1.0, kRadius / abs(P.z - sceneZ));
            occlusion += range;
        }
    }
    float ao = 1.0 - (occlusion / 16.0);
    ao = pow(clamp(ao, 0.0, 1.0), 1.6); // contrast: crisper crevices
    outAoDepth = vec2(ao, P.z);
}
