#version 460 core

// Screen-space global illumination: one bounce of indirect light. Same depth-reconstruct
// + hemisphere-sample skeleton as SSAO, but instead of counting occlusion it gathers the
// lit colour of nearby surfaces, weighted by cosine and distance -> coloured fill light
// and colour bleeding (green under trees, warm off lit stone). Half res; the kernel
// rotates per frame so TAA integrates the noise into a smooth result.

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uDepth;
layout(binding = 1) uniform sampler2D uScene; // lit HDR scene to bounce
layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform float uTime;

const vec3 kKernel[16] = vec3[16](
    vec3( 0.53,  0.10,  0.30), vec3(-0.28,  0.36,  0.21), vec3( 0.10, -0.58,  0.45),
    vec3(-0.62, -0.15,  0.37), vec3( 0.21,  0.44,  0.63), vec3(-0.11, -0.20,  0.19),
    vec3( 0.35, -0.29,  0.71), vec3(-0.44,  0.52,  0.48), vec3( 0.66,  0.31,  0.55),
    vec3(-0.19,  0.08,  0.09), vec3( 0.05,  0.64,  0.30), vec3(-0.35, -0.51,  0.60),
    vec3( 0.48, -0.44,  0.24), vec3(-0.57,  0.22,  0.66), vec3( 0.14,  0.71,  0.12),
    vec3(-0.08, -0.36,  0.42));

float rawToViewZ(float d) { return -uNear / max(d, 1e-7); }
vec3 viewPos(vec2 uv) {
    float z = rawToViewZ(texture(uDepth, uv).r);
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * uProjScale.x * -z, ndc.y * uProjScale.y * -z, z);
}
vec2 project(vec3 p) {
    return vec2(p.x / (uProjScale.x * -p.z), p.y / (uProjScale.y * -p.z)) * 0.5 + 0.5;
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec3 P = viewPos(vUv);
    if (P.z < -400.0) { // sky: no bounce
        outColor = vec4(0.0);
        return;
    }
    // reconstruct the face normal from depth derivatives. bail on the degenerate case where
    // the 2x2 quad straddles a depth cliff (e.g. terrain edge against the sky): the two
    // derivatives go near-parallel, the cross collapses, and normalize() would NaN. the
    // !(x > eps) form also rejects a NaN cross outright.
    vec3 cr = cross(dFdx(P), dFdy(P));
    if (!(dot(cr, cr) > 1e-12)) {
        outColor = vec4(0.0);
        return;
    }
    vec3 N = normalize(cr);

    float a = (hash(gl_FragCoord.xy) + uTime * 0.61) * 6.2831853; // per-pixel + per-frame
    vec3 rot = vec3(cos(a), sin(a), 0.0);
    vec3 T = normalize(rot - N * dot(rot, N));
    vec3 B = cross(N, T);

    const float kRadius = 2.2;
    vec3 gi = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        vec3 s = P + (T * kKernel[i].x + B * kKernel[i].y + N * kKernel[i].z) * kRadius;
        vec2 suv = project(s);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0)
            continue;
        vec3 d = viewPos(suv) - P;
        float dist = length(d);
        if (dist < 0.05 || dist > kRadius)
            continue;
        float ndl = max(dot(N, d / dist), 0.0);      // sample in the hemisphere
        float fall = 1.0 - dist / kRadius;
        gi += texture(uScene, suv).rgb * ndl * fall * fall;
    }
    gi /= 16.0;
    if (any(isnan(gi)) || any(isinf(gi))) // never let a stray NaN poison the composite
        gi = vec3(0.0);
    outColor = vec4(clamp(gi, vec3(0.0), vec3(8.0)), 1.0);
}
