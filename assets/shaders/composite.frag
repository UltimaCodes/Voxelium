#version 460 core

// applies SSAO to the scene, paints a real sky for pixels with no geometry (gradient +
// sun disc + horizon glow — no extra pass), and adds volumetric sun scattering
// raymarched through the shadow cascades for geometry pixels. output stays HDR; bloom
// and tonemapping come after.

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uScene;
layout(binding = 2) uniform sampler2D uAo;
layout(binding = 3) uniform sampler2D uDepth;
layout(binding = 4) uniform sampler2D uSsgi;
layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;

layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot; // view -> camera-relative world; eats 2,3,4
layout(location = 5) uniform vec3 uSunL;
layout(location = 6) uniform mat4 uCascadeVp[3];
layout(location = 18) uniform vec3 uCascadeSplit;
layout(location = 19) uniform float uUnderwater; // 1 when the camera eye is submerged
layout(location = 20) uniform float uCloudsOn;
layout(location = 21) uniform float uTime;
layout(location = 22) uniform float uSsgiOn;
layout(location = 23) uniform float uAoOn;         // moved off 3 (was aliased by the mat3)
layout(location = 24) uniform float uVolumetricOn; // moved off 4
layout(location = 25) uniform vec3 uSunColor;      // day/night sun radiance

const vec3 kWaterDeep = vec3(0.03, 0.13, 0.20);

vec3 viewRay() {
    vec2 ndc = vUv * 2.0 - 1.0;
    return normalize(uInvViewRot * vec3(ndc.x * uProjScale.x, ndc.y * uProjScale.y, -1.0));
}

float starHash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec3 skyColor(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    float alt = uSunL.y;                             // sun altitude, -1..1
    float day = clamp(alt * 3.0 + 0.12, 0.0, 1.0);   // 1 day, 0 night
    // day/night gradient
    vec3 zenith = mix(vec3(0.01, 0.015, 0.05), vec3(0.14, 0.34, 0.80), day);
    vec3 horizon = mix(vec3(0.04, 0.06, 0.12), vec3(0.66, 0.78, 0.92), day);
    vec3 col = mix(horizon, zenith, pow(up, 0.5));
    if (dir.y < 0.0)
        col = mix(horizon, mix(vec3(0.06, 0.07, 0.09), vec3(0.34, 0.38, 0.44), day),
                  clamp(-dir.y * 3.0, 0.0, 1.0));

    float sd = max(dot(dir, uSunL), 0.0);
    float above = smoothstep(-0.03, 0.06, alt);
    float lowSun = clamp(1.0 - abs(alt) * 2.2, 0.0, 1.0) * day; // sunrise/sunset warmth
    col += vec3(1.0, 0.34, 0.10) * pow(sd, 3.0) * lowSun * 2.4;            // warm band
    col = mix(col, col * vec3(1.2, 0.92, 0.72), lowSun * 0.5);            // golden-hour cast
    col += vec3(1.0, 0.55, 0.28) * pow(sd, 8.0) * 0.6 * above;             // glow
    col += uSunColor * smoothstep(0.9986, 0.9995, sd) * 3.2 * above;       // sun disc

    float night = 1.0 - day;
    if (night > 0.01) {
        vec2 g = floor(dir.xy * 130.0 + dir.z * 41.0);
        float star = step(0.992, starHash(g)) * step(0.0, dir.y);
        col += vec3(0.9, 0.92, 1.0) * star * night * 0.9;
        float md = max(dot(dir, -uSunL), 0.0);                            // moon opposite sun
        col += vec3(0.75, 0.78, 0.9) * smoothstep(0.9990, 0.9997, md) * night * 2.2;
        col += vec3(0.2, 0.24, 0.36) * pow(md, 250.0) * night;            // moon halo
    }
    return col;
}

// drifting fbm cloud dome: flatten the view direction onto a plane (perspective toward the
// horizon), fbm coverage, and a cheap sun-side density difference for shading + silver
// lining. smooth noise -> soft organic clouds, never blocky.
float vhash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(vhash(i), vhash(i + vec2(1, 0)), f.x),
               mix(vhash(i + vec2(0, 1)), vhash(i + vec2(1, 1)), f.x), f.y);
}
float cloudFbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; ++i) {
        s += a * vnoise(p);
        p = p * 2.02 + 3.1;
        a *= 0.5;
    }
    return s;
}
vec4 clouds(vec3 dir) {
    if (dir.y < 0.03)
        return vec4(0.0);
    vec2 p = dir.xz / dir.y * 0.6 + vec2(uTime * 0.008, uTime * 0.005); // flatten + wind
    float cov = cloudFbm(p);
    float d = smoothstep(0.42, 0.66, cov) * smoothstep(0.03, 0.22, dir.y);
    if (d <= 0.001)
        return vec4(0.0);
    vec2 sp = (uSunL.xz / max(abs(uSunL.y), 0.25)) * 0.12; // one step toward the sun
    float shade = clamp((cov - cloudFbm(p + sp)) * 2.2 + 0.55, 0.25, 1.0);
    vec3 col = mix(vec3(0.52, 0.56, 0.64), vec3(1.05, 1.02, 0.98), shade);
    float day = clamp(uSunL.y * 3.0 + 0.12, 0.0, 1.0);
    col += vec3(1.0, 0.55, 0.2) * pow(max(dot(dir, uSunL), 0.0), 6.0) * day; // warm sun side
    col *= 0.16 + 0.84 * day; // clouds go dim + cool at night (moonlit, not sunlit)
    return vec4(col, d);
}
vec3 skyWithClouds(vec3 dir) {
    vec3 s = skyColor(dir);
    if (uCloudsOn > 0.5) {
        vec4 c = clouds(dir);
        s = mix(s, c.rgb, c.a);
    }
    return s;
}

// 4-tap softened shadow lookup for the raymarch — a single tap makes the light shafts
// alias into a grid against the shadow texels; averaging a few taps smooths them out.
float shadowSingle(vec3 pos) {
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    vec4 sc = uCascadeVp[c] * vec4(pos, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float z = sc.z - 0.0015;
    float t = 1.5 / 1536.0;
    return 0.25 * (texture(uShadowMap, vec4(uv + vec2(t, t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(-t, t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(t, -t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(-t, -t), float(c), z)));
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// interleaved gradient noise — a much better dither than a plain hash, so the raymarch
// start offset breaks up banding evenly instead of in blotches
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

void main() {
    float rawD = texture(uDepth, vUv).r;
    vec3 dir = viewRay();

    // sky pixel (reversed-z far = 0): paint the sky instead of the flat clear colour.
    // underwater there is no sky — the far distance is just deep water.
    if (rawD <= 0.0) {
        outColor = vec4(uUnderwater > 0.5 ? kWaterDeep : skyWithClouds(dir), 1.0);
        return;
    }

    vec3 color = texture(uScene, vUv).rgb;
    if (uAoOn > 0.5)
        color *= texture(uAo, vUv).r;
    if (uSsgiOn > 0.5)
        color += texture(uSsgi, vUv).rgb * 0.9; // one-bounce indirect fill

    if (uVolumetricOn > 0.5) {
        float viewZ = -uNear / max(rawD, 1e-7);
        vec2 ndc = vUv * 2.0 - 1.0;
        vec3 vpos = vec3(ndc.x * uProjScale.x * -viewZ, ndc.y * uProjScale.y * -viewZ, viewZ);
        vec3 world = uInvViewRot * vpos;
        float len = min(length(world), 150.0);
        vec3 rd = normalize(world);

        const int kSteps = 16;
        float stepLen = len / float(kSteps);
        float t = stepLen * ign(gl_FragCoord.xy); // dithered start, breaks up banding
        float scatter = 0.0;
        for (int i = 0; i < kSteps; ++i) {
            scatter += shadowSingle(rd * t);
            t += stepLen;
        }
        scatter /= float(kSteps);
        // stronger forward lobe + a touch more intensity so the god-ray shafts read
        // clearly when looking toward the sun, without bringing the aliased grid back
        float phase = 0.10 + 0.90 * pow(max(dot(rd, uSunL), 0.0), 5.0);
        // uSunColor carries the day/night intensity, so the shafts fade out at night
        color += uSunColor * (scatter * phase * len * 0.00018);
    }

    // underwater: absorb reds and fog toward deep water with distance (Beer-Lambert-ish)
    if (uUnderwater > 0.5) {
        float viewZ = -uNear / max(rawD, 1e-7);
        vec2 ndc = vUv * 2.0 - 1.0;
        vec3 vpos = vec3(ndc.x * uProjScale.x * -viewZ, ndc.y * uProjScale.y * -viewZ, viewZ);
        float dist = length(vpos);
        color *= vec3(0.55, 0.82, 1.0);            // water eats warm light first
        color = mix(color, kWaterDeep, 1.0 - exp(-dist * 0.10));
    }

    outColor = vec4(color, 1.0);
}
