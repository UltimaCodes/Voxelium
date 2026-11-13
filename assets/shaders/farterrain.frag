#version 460 core

// Lit by the same model as the chunks, its shadows marched across its own height texture, and a
// little procedural surface (tone, tree clumps) so a grid cell does not read as a smooth blob. The
// composite pass melts it into the sky with the same distance fog as everything else.
in vec3 vColor;
in vec3 vNormal;
in vec3 vRel;
in vec2 vWorld;
flat in float vWater;
in float vAo;
layout(location = 0) out vec4 outColor;

layout(location = 5) uniform float uCamY;
layout(location = 6) uniform float uInner;
layout(location = 7) uniform vec3 uSunL;
layout(location = 8) uniform vec3 uSunColor;
layout(location = 9) uniform vec3 uSkyAmbient;
layout(location = 13) uniform vec3 uLandSun;
layout(location = 14) uniform vec3 uLandAmb;
layout(location = 15) uniform vec3 uCamPos;
layout(location = 16) uniform float uCloudShadow;
layout(location = 17) uniform vec3 uWaterLevel;
layout(location = 18) uniform vec3 uHole;      // x, z of the middle of what is drawn nearer in, half its width
layout(location = 19) uniform vec3 uHeightXf;  // x, z of this ring's corner, 1 / its span
layout(location = 31) uniform float uCloudTime;
layout(binding = 8) uniform sampler2D uHeights;

#include "clouds.glsl"
#define CLOUD_WEATHER_BINDING 6
#include "cloudfield.glsl"
#include "lighting.glsl"

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

float screenNoise() { return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))); }

// Whether the sun reaches this point past the hills between it and the sun: walk toward it over the
// height texture, in growing steps, softening with distance.
float hillShadow(vec3 wp) {
    if (uSunL.y <= 0.02)
        return 1.0;
    const int kSteps = 12;
    float lit = 1.0;
    float t = 8.0 + 24.0 * screenNoise();
    float stepLen = 24.0;
    for (int i = 0; i < kSteps; ++i) {
        vec3 p = wp + uSunL * t;
        vec2 uv = (p.xz - uHeightXf.xy) * uHeightXf.z;
        float h = textureLod(uHeights, uv, 0.0).r;
        float soft = 2.0 + t * 0.018;
        lit = min(lit, smoothstep(-soft, soft, p.y - h));
        t += stepLen;
        stepLen *= 1.32;
    }
    return lit;
}

// The sky a low glance at the water reflects, as the near water's own.
vec3 waterSky(vec3 d, float day) {
    float up = clamp(d.y, 0.0, 1.0);
    return mix(vec3(0.66, 0.78, 0.92), vec3(0.14, 0.34, 0.80), pow(up, 0.5)) * day + vec3(0.008, 0.012, 0.024) * (1.0 - day);
}

void main() {
    // the streamed chunks own everything inside this radius, a finer ring everything in its square
    if (length(vRel.xz) < uInner - 24.0)
        discard;
    if (uHole.z > 0.0 && max(abs(vWorld.x - uHole.x), abs(vWorld.y - uHole.y)) < uHole.z)
        discard;
    vec3 N = normalize(vNormal);
    if (vWater > 0.5) {
        // Where the streamed sea is still over it, come in gradually, a few pixels more each step
        // out (TAA smooths the dither): the near sea's floor shows through less and less, instead of
        // ending on a hard circle with the reefs cut off along it. Whole by the edge of the chunks,
        // which reach uInner / 0.92 (fogEnd is 92% of their radius).
        float d = length(vRel.xz);
        float cover = smoothstep(uInner - 24.0, uInner / 0.92 - 4.0, d);
        float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy + fract(uCloudTime * 7.31) * 61.0,
                                                vec2(0.06711056, 0.00583715))));
        if (n >= cover)
            discard;
        // the same parts as the near water: a body colour by depth, and the sky by Fresnel
        float absorb = 1.0 - exp(-vColor.r * 40.0 * 0.18);
        vec3 tint = mix(vec3(0.10, 0.26, 0.40), vec3(0.015, 0.06, 0.18), absorb);
        float day = smoothstep(0.0, 1.0, clamp(uSkyAmbient.b / 0.52, 0.0, 1.0));
        vec3 body = tint * (uSkyAmbient + uSunColor * max(uSunL.y, 0.0) * 0.35) * uWaterLevel;
        vec3 V = normalize(-vRel);
        float fres = mix(0.04, 1.0, pow(1.0 - max(V.y, 0.0), 5.0));
        vec3 refl = waterSky(reflect(-V, vec3(0.0, 1.0, 0.0)), day) * uWaterLevel;
        float glint = pow(max(dot(reflect(-uSunL, vec3(0.0, 1.0, 0.0)), V), 0.0), 90.0);
        // alpha 0 tells the near water, where it overlaps, this is no sea floor
        outColor = vec4(mix(body, refl, fres) + uSunColor * glint * 0.22, 0.0);
        return;
    }
    float n = fbm(vWorld * 0.045);
    float n2 = fbm(vWorld * 0.19);
    vec3 albedo = pow(vColor, vec3(2.2)) * (0.86 + 0.28 * n + 0.10 * (n2 - 0.5));
    float green = smoothstep(0.02, 0.10, vColor.g - max(vColor.r, vColor.b));
    float clump = smoothstep(0.52, 0.72, fbm(vWorld * 0.11 + 7.0));
    albedo = mix(albedo, albedo * vec3(0.55, 0.68, 0.52), clump * green * 0.75);

    // Near the chunks, whose top faces all look straight up, the slope barely shows. Further off the
    // hills take on their full shape.
    float dist = length(vRel.xz);
    N = normalize(mix(vec3(0.0, 1.0, 0.0), N + vec3(n - 0.5, 0.0, n2 - 0.5) * 0.3, smoothstep(uInner, uInner + 1200.0, dist)));

    vec3 wp = vec3(uCamPos.x + vRel.x, uCamY + vRel.y, uCamPos.z + vRel.z);
    float cloud = cloudBlock(wp, uSunL, uCloudShadow);
    float shadow = hillShadow(wp);
    vec3 sun = uLandSun * (max(dot(N, uSunL), 0.0) * shadow * (1.0 - cloud));
    vec3 amb = (skyFill(N, uLandAmb) * (1.0 + 0.2 * cloud) + sunFill(N, uSunL, uLandSun)) * vAo;
    outColor = vec4(albedo * (sun + amb), 1.0);
}
