#version 460 core

// Lit and coloured: no shadows and no real texture, but a little procedural surface (tone, tree
// clumps, rock on slopes) so a 64-block grid does not read as smooth blobs. It is only ever seen
// through haze, and the composite pass melts it into the sky with the same distance fog as
// everything else.
in vec3 vColor;
in vec3 vNormal;
in vec3 vRel;
in vec2 vWorld;
flat in float vWater;
layout(location = 0) out vec4 outColor;

layout(location = 6) uniform float uInner;
layout(location = 7) uniform vec3 uSunL;
layout(location = 8) uniform vec3 uSunColor;
layout(location = 9) uniform vec3 uSkyAmbient;

float h21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vn(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) { return 0.55 * vn(p) + 0.30 * vn(p * 2.03 + 5.1) + 0.15 * vn(p * 4.1 + 9.7); }

void main() {
    // the streamed chunks own everything inside this radius
    if (length(vRel.xz) < uInner - 24.0)
        discard;
    vec3 N = normalize(vNormal);
    vec3 albedo = vColor;
    if (vWater < 0.5) {
        // The grid is 64 blocks a cell, far too coarse to have any character of its own, so the
        // surface is given some: broad tone, tree clumps on green ground, bare rock where it is
        // steep, and a rough normal so the light catches ridges instead of gliding over them.
        float n = fbm(vWorld * 0.045);
        float n2 = fbm(vWorld * 0.19);
        albedo *= 0.86 + 0.28 * n + 0.10 * (n2 - 0.5);
        float green = smoothstep(0.02, 0.10, vColor.g - max(vColor.r, vColor.b));
        float clump = smoothstep(0.52, 0.72, fbm(vWorld * 0.11 + 7.0));
        albedo = mix(albedo, albedo * vec3(0.52, 0.72, 0.50), clump * green * 0.75);
        float rock = smoothstep(0.34, 0.60, 1.0 - N.y);
        albedo = mix(albedo, vec3(0.44, 0.42, 0.42) * (0.75 + 0.5 * n2), rock * (1.0 - step(0.85, vColor.b)));
        N = normalize(N + vec3(n - 0.5, 0.0, n2 - 0.5) * 0.55);
    }
    float ndl = max(dot(N, uSunL), 0.0);
    vec3 col = albedo * (uSkyAmbient * 1.15 + uSunColor * ndl * 0.85);
    if (vWater > 0.5) {
        vec3 V = normalize(-vRel);
        float glint = pow(max(dot(reflect(-uSunL, vec3(0.0, 1.0, 0.0)), V), 0.0), 90.0);
        col += uSunColor * glint * 0.22;
        float fres = pow(1.0 - clamp(V.y, 0.0, 1.0), 3.0);
        col = mix(col, uSkyAmbient * 1.6, 0.35 * fres);
    }
    outColor = vec4(col, 1.0);
}
