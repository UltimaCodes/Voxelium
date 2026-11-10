#version 460 core

// Real clouds, marched through a volume at half resolution, a quarter of the pixels a frame, the rest
// carried over from the last frame through the camera's movement. So a march gets the steps it needs
// and still costs about a ms. rgb is the light the cloud sends this way, a how much of what is behind
// it gets through.
#ifndef VX_QUALITY
#define VX_QUALITY 2
#endif

in vec2 vUv;
layout(location = 0) out vec4 outCloud;

layout(binding = 3) uniform sampler2D uDepth;
layout(binding = 4) uniform sampler2D uHistory;
layout(binding = 12) uniform sampler3D uShape;
layout(binding = 13) uniform sampler3D uDetail;

layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot;     // eats 2,3,4
layout(location = 5) uniform mat3 uPrevInvViewRot; // eats 5,6,7
layout(location = 8) uniform vec3 uCamDelta;       // how far the camera moved since last frame
layout(location = 9) uniform vec3 uCamPos;
layout(location = 10) uniform float uTime;
layout(location = 11) uniform vec3 uLightL;        // sun by day, moon by night
layout(location = 12) uniform vec3 uLightColor;
layout(location = 13) uniform vec3 uSkyAmbient;
layout(location = 14) uniform float uDaylight;
layout(location = 15) uniform float uFlash;
layout(location = 16) uniform int uFrame;
layout(location = 17) uniform float uHistoryOk;
layout(location = 18) uniform vec3 uSunL;
layout(location = 19) uniform float uOvercast;

#include "clouds.glsl"
#include "cloudfield.glsl"
#include "sky.glsl"

const float kShapeTile = 560.0, kDetailTile = 92.0;
const float kExtinction = 0.095; // per block at full density

#if VX_QUALITY <= 1
const int kSteps = 20;
const int kLightSteps = 2;
#elif VX_QUALITY == 2
const int kSteps = 28;
const int kLightSteps = 3;
#else
const int kSteps = 36;
const int kLightSteps = 4;
#endif

float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

float density(vec3 p, float lod, bool detailed) {
    vec2 w = cloudWeather(p.xz, uTime);
    float cover = cloudCoverage(w.r);
    if (cover <= 0.0)
        return 0.0;
    float y = (p.y - kCloudBase) / (kCloudTop - kCloudBase);
    // a flat base and a rounded head, taller where the weather map says so
    float top = mix(0.34, 1.0, smoothstep(0.32, 0.72, w.g));
    float shape = smoothstep(0.0, 0.07, y) * (1.0 - smoothstep(top * 0.55, top, y));
    if (shape <= 0.0)
        return 0.0;
    vec3 sp = (p + vec3(kCloudWind.x * uTime * 1.25, uTime * 0.5, kCloudWind.y * uTime * 1.25)) / kShapeTile;
    float base = smoothstep(0.34, 0.90, textureLod(uShape, sp, lod).r) * shape;
    // Even full cover leaves the noise something to carve: the threshold never reaches zero, so a big
    // cloud is billows and gaps, and not one smooth mass.
    float cut = 1.0 - cover * 0.60;
    float c = clamp((base - cut) / max(1.0 - cut, 1e-3), 0.0, 1.0);
    // frayed where it is thin; deep inside a cloud nobody can see the difference
    if (detailed && c > 0.0 && c < 0.55) {
        vec3 dp = (p + vec3(kCloudWind.x * uTime * 2.0, -uTime * 1.2, kCloudWind.y * uTime * 2.0)) / kDetailTile;
        float e = textureLod(uDetail, dp, max(lod - 1.0, 0.0)).r * 0.46;
        c = clamp((c - e) / (1.0 - e), 0.0, 1.0);
    }
    return c;
}

float hg(float cosA, float g) {
    float g2 = g * g;
    return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cosA, 1.5) * 0.0796;
}

vec4 march(vec3 dir, float maxDist) {
    float y0 = uCamPos.y;
    float t0, t1;
    if (y0 < kCloudBase) {
        if (dir.y <= 0.01)
            return vec4(0.0, 0.0, 0.0, 1.0);
        t0 = (kCloudBase - y0) / dir.y;
        t1 = (kCloudTop - y0) / dir.y;
    } else if (y0 > kCloudTop) {
        if (dir.y >= -0.01)
            return vec4(0.0, 0.0, 0.0, 1.0);
        t0 = (kCloudTop - y0) / dir.y;
        t1 = (kCloudBase - y0) / dir.y;
    } else {
        t0 = 0.0;
        t1 = dir.y > 0.001 ? (kCloudTop - y0) / dir.y : (dir.y < -0.001 ? (kCloudBase - y0) / dir.y : 9000.0);
    }
    const float kFar = 16000.0;
    if (t0 > kFar || t0 > maxDist)
        return vec4(0.0, 0.0, 0.0, 1.0);
    t1 = min(min(t1, t0 + 5200.0), maxDist);

    float stepLen = (t1 - t0) / float(kSteps);
    float t = t0 + stepLen * ign(gl_FragCoord.xy + 5.588238 * float(uFrame & 15));
    float lod = clamp(log2(max(stepLen, 1.0) / 30.0), 0.0, 2.5);

    float cosA = dot(dir, uLightL);
    // a bright rim looking toward the light, and a little light thrown back the other way
    float phase = mix(hg(cosA, 0.72), hg(cosA, -0.18), 0.32);
    float phase2 = mix(hg(cosA, 0.36), hg(cosA, -0.10), 0.5);

    // rain clouds are thick enough that the sun hardly gets through, and the sky above them is grey
    vec3 ambientTop = uSkyAmbient * mix(2.1, 1.05, uOvercast) + vec3(0.75, 0.84, 1.05) * (0.9 * uFlash);
    vec3 lightColor = uLightColor * mix(1.0, 0.16, uOvercast);
    float T = 1.0;
    vec3 light = vec3(0.0);
    float firstHit = -1.0;
    for (int i = 0; i < kSteps; ++i) {
        vec3 p = uCamPos + dir * t;
        float d = density(p, lod, true);
        if (d > 0.002) {
            if (firstHit < 0.0)
                firstHit = t;
            // how much cloud between here and the light, in growing steps
            float od = 0.0;
            float ls = 13.0, lt = 7.0;
            for (int j = 0; j < kLightSteps; ++j) {
                od += density(p + uLightL * lt, lod + 1.0, false) * ls;
                lt += ls * 1.5;
                ls *= 2.1;
            }
            od *= kExtinction * (uLightL.y > 0.0 ? 1.0 : 0.6);
            // light that has scattered inside gets through further than Beer alone lets it
            float through = exp(-od) * 0.75 + exp(-od * 0.25) * 0.25;
            // the edge of a thick cloud is dark seen from the sun's side
            float powder = 1.0 - 0.55 * exp(-d * 9.0 - od * 0.6);
            float y = clamp((p.y - kCloudBase) / (kCloudTop - kCloudBase), 0.0, 1.0);
            vec3 sun = lightColor * (through * powder * (phase * 8.5 + phase2 * 7.5 * exp(-od * 0.5)));
            vec3 amb = ambientTop * (0.42 + 0.58 * y);
            float ext = d * kExtinction;
            float a = exp(-ext * stepLen);
            // energy conserving over the step, so step length does not change the brightness
            light += T * (sun + amb) * (1.0 - a);
            T *= a;
            if (T < 0.015)
                break;
        }
        t += stepLen;
    }
    if (firstHit < 0.0)
        return vec4(0.0, 0.0, 0.0, 1.0);
    // At night a cloud is a dark shape that hides the stars, darker than the sky around it, with
    // only its moonward edge picked out. Full light comes back by the time dusk is underway.
    light *= mix(0.3, 1.0, smoothstep(0.0, 0.25, uDaylight));
    // Far off, a cloud is mostly the air in front of it, and they fade out before the horizon.
    vec3 flatDir = normalize(vec3(dir.x, max(dir.y, 0.02), dir.z));
    vec3 haze = skyGradient(flatDir, uSunL, uDaylight, uOvercast);
    float air = 1.0 - exp(-firstHit * 0.00011);
    float fade = y0 < kCloudBase ? smoothstep(0.012, 0.09, dir.y) : 1.0;
    float alpha = (1.0 - T) * fade;
    light = mix(light, haze * (1.0 - T), air * 0.8) * fade;
    return vec4(light, 1.0 - alpha);
}

// Thin high ice cloud, a long way above the rest and combed by the wind. Painted on a plane.
vec4 cirrus(vec4 under, vec3 dir) {
    if (dir.y < 0.03 || uCamPos.y > 1500.0)
        return under;
    vec2 q = uCamPos.xz + dir.xz * ((1500.0 - uCamPos.y) / dir.y) + kCloudWind * uTime * 3.0;
    // long along the wind, narrow across it
    vec2 w = normalize(kCloudWind);
    vec2 uvw = vec2(dot(q, w) / 14000.0, dot(q, vec2(-w.y, w.x)) / 3200.0);
    float n = textureLod(uShape, vec3(uvw, 0.31), 1.0).r;
    n = n * 0.72 + textureLod(uShape, vec3(uvw * 3.1, 0.67), 0.0).r * 0.28;
    // in patches, and most of the sky has none
    float patch_ = smoothstep(0.52, 0.72, textureLod(uCloudWeather, q / 21000.0, 0.0).g);
    float a = smoothstep(0.62, 0.88, n) * patch_ * smoothstep(0.03, 0.22, dir.y) * 0.20 * (1.0 - uOvercast);
    if (a <= 0.0)
        return under;
    vec3 col = (uLightColor * (0.55 + 2.2 * hg(dot(dir, uLightL), 0.6)) + uSkyAmbient * 1.2) * a;
    return vec4(under.rgb + under.a * col, under.a * (1.0 - a));
}

void main() {
    vec2 ndc = vUv * 2.0 - 1.0;
    vec3 dir = normalize(uInvViewRot * vec3(ndc.x * uProjScale.x, ndc.y * uProjScale.y, -1.0));
    float y0 = uCamPos.y;
    float entry = y0 < kCloudBase ? (dir.y > 0.01 ? (kCloudBase - y0) / dir.y : 0.0) : 0.0;

    // Where this pixel's cloud was last frame, taking it to sit where the ray enters the layer.
    vec3 was = transpose(uPrevInvViewRot) * (dir * max(entry + 60.0, 200.0) + uCamDelta);
    vec2 prevUv = vec2(was.x / (uProjScale.x * -was.z), was.y / (uProjScale.y * -was.z)) * 0.5 + 0.5;
    bool haveHistory = uHistoryOk > 0.5 && was.z < 0.0 && all(greaterThanEqual(prevUv, vec2(0.0))) && all(lessThanEqual(prevUv, vec2(1.0)));

    ivec2 ip = ivec2(gl_FragCoord.xy);
    const int kOrder[4] = int[4](0, 3, 1, 2);
    bool update = !haveHistory || ((ip.x & 1) + 2 * (ip.y & 1)) == kOrder[uFrame & 3];
    if (!update) {
        outCloud = texture(uHistory, prevUv);
        return;
    }
    // Below the layer the composite decides per full-res pixel what stands in front of it, so the
    // silhouette of a mountain has no halo. Inside or above it the ground has to stop the ray here.
    float maxDist = 1e9;
    if (y0 >= kCloudBase) {
        float rawD = textureLod(uDepth, vUv, 0.0).r;
        if (rawD > 0.0)
            maxDist = (uNear / rawD) / max(dot(dir, uInvViewRot * vec3(0.0, 0.0, -1.0)), 0.05);
    }
    vec4 now = cirrus(march(dir, maxDist), dir);
    outCloud = haveHistory ? mix(texture(uHistory, prevUv), now, 0.55) : now;
}
