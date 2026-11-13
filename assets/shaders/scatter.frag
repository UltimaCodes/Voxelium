#version 460 core

// Injected by Shader.cpp from the active quality preset; see composite.frag for why these are
// defines and not uniforms.
#ifndef VX_QUALITY
#define VX_QUALITY 2
#endif
#ifndef VX_SHADOW_RES
#define VX_SHADOW_RES 1536.0
#endif

// Light in the air, at half resolution: the sun's shafts marched through the shadow cascades
// and lamp light marched through the light volume, summed into one texture the composite reads
// back up (depth-aware, see upsampleScatter there). The two marches were most of what the
// composite cost, on Ultra 24 steps of four shadow taps a pixel, and the light they make is
// soft enough that a quarter of the pixels carries it.

in vec2 vUv;
layout(location = 0) out vec4 outScatter; // rgb = light the air scatters back, a = distance marched to

layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;
layout(binding = 3) uniform sampler2D uDepth;
layout(binding = 7) uniform sampler3D uLightVol; // voxel light around the camera (LightVolume)

layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot; // view -> camera-relative world; eats 2,3,4
layout(location = 6) uniform mat4 uCascadeVp[3];
layout(location = 18) uniform vec3 uCascadeSplit;
layout(location = 24) uniform float uVolumetricOn; // sun shafts
layout(location = 30) uniform vec3 uLightL;        // toward sun by day, moon by night
layout(location = 31) uniform vec3 uLightColor;    // radiance of uLightL
layout(location = 37) uniform vec3 uVolCam;        // camera in light-volume texels, axis order (x, z, y)
layout(location = 38) uniform vec3 uVolMin;        // the volume's low corner relative to the camera
layout(location = 39) uniform float uVolOn;        // 1 when some block light is in range
layout(location = 40) uniform float uShaftGain;    // 0 with no sky over the camera
layout(location = 41) uniform float uUnderwater;   // 1 when the camera is under water
layout(location = 42) uniform vec3 uCamPos;        // camera world position
layout(location = 43) uniform float uTime;

// How much a block of air scatters back of the lamp light in it. Tuned against hidden cave
// shots so a lamp gets a halo a few blocks across without fogging the cave.
const float kLampAir = 0.03;

#include "march.glsl"
#include "clouds.glsl"
#define CLOUD_WEATHER_BINDING 6
#include "cloudfield.glsl"
layout(location = 44) uniform float uCloudTime;
layout(location = 45) uniform float uCloudsOn;
#include "lighting.glsl"

// How much of the sun gets past the clouds to a point: a shaft needs a gap to come through.
float sunThroughCloud(vec3 worldPos) {
    return uCloudsOn > 0.5 ? 1.0 - 0.9 * cloudCover(worldPos, uLightL) : 1.0;
}

const float kSeaLevel = 62.0;

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

// The light under water comes down through the surface in moving sheets. Each point on the ray asks
// where its light crossed the surface and whether the waves were letting it through there.
vec3 waterShafts(vec3 rd, float dist) {
    if (uLightL.y <= 0.05)
        return vec3(0.0);
    // Refraction bends it toward straight down.
    vec3 L = normalize(vec3(uLightL.x * 0.75, uLightL.y, uLightL.z * 0.75));
#if VX_QUALITY <= 1
    const int kSteps = 10;
#else
    const int kSteps = 18;
#endif
    float len = min(dist, 80.0);
    float stepLen = len / float(kSteps);
    float t = stepLen * ign(gl_FragCoord.xy);
    vec3 sum = vec3(0.0);
    for (int i = 0; i < kSteps; ++i) {
        vec3 rel = rd * t;
        vec3 p = uCamPos + rel;
        float below = kSeaLevel - p.y;
        if (below > 0.0) {
            vec2 hit = p.xz + L.xz / L.y * below;
            // thin moving streaks where the waves focus the light, not broad sheets
            float n = 0.6 * vn(hit * 0.22 + uTime * vec2(0.11, 0.06)) + 0.4 * vn(hit * 0.57 - uTime * vec2(0.15, 0.09));
            float sheet = smoothstep(0.56, 0.74, n);
            // the water it came down through and the water between it and the eye, red first
            vec3 lost = exp(-vec3(0.22, 0.075, 0.03) * (below / L.y + t));
            sum += sheet * shadowSingle(rel) * lost * sunThroughCloud(vec3(hit.x, kSeaLevel, hit.y));
        }
        t += stepLen;
    }
    // mostly toward the light, faint across it
    float phase = 0.12 + 0.88 * pow(max(dot(rd, L), 0.0), 4.0);
    return uLightColor * (sum * stepLen * phase * 0.03);
}

void main() {
    // One full-res depth texel stands for the four this pixel covers; the composite's
    // upsample sorts out the edges where they disagree.
    ivec2 full = ivec2(gl_FragCoord.xy) * 2;
    ivec2 fullSize = textureSize(uDepth, 0);
    full = min(full, fullSize - 1);
    float rawD = texelFetch(uDepth, full, 0).r;
    vec2 uv = (vec2(full) + 0.5) / vec2(fullSize);
    if (rawD <= 0.0) { // sky: the composite paints it without reading this, unless it is open water
        vec3 water = vec3(0.0);
        if (uUnderwater > 0.5 && uVolumetricOn > 0.5) {
            vec2 skyNdc = uv * 2.0 - 1.0;
            water = waterShafts(normalize(uInvViewRot * vec3(skyNdc.x * uProjScale.x, skyNdc.y * uProjScale.y, -1.0)), 80.0);
        }
        outScatter = vec4(water * uShaftGain, 60000.0);
        return;
    }
    float viewZ = -uNear / max(rawD, 1e-7);
    vec2 ndc = uv * 2.0 - 1.0;
    vec3 worldRel = uInvViewRot *
                    vec3(ndc.x * uProjScale.x * -viewZ, ndc.y * uProjScale.y * -viewZ, viewZ);
    float dist = length(worldRel);
    vec3 rd = worldRel / max(dist, 1e-6);
    vec3 light = vec3(0.0);

    if (uUnderwater > 0.5) {
        if (uVolumetricOn > 0.5)
            light += waterShafts(rd, dist) * uShaftGain;
    } else if (uVolumetricOn > 0.5) {
        float len = min(dist, 150.0);
#if   VX_QUALITY == 0
        const int kSteps = 6;
#elif VX_QUALITY == 1
        const int kSteps = 8;
#elif VX_QUALITY == 3
        const int kSteps = 24;
#else
        const int kSteps = 16;
#endif
        float stepLen = len / float(kSteps);
        float t = stepLen * ign(gl_FragCoord.xy); // dithered start, breaks up banding
        float scatter = 0.0;
        for (int i = 0; i < kSteps; ++i) {
            scatter += shadowSingle(rd * t) * sunThroughCloud(uCamPos + rd * t);
            t += stepLen;
        }
        scatter /= float(kSteps);
        // stronger forward lobe + a touch more intensity so the god-ray shafts read
        // clearly when looking toward the sun, without bringing the aliased grid back
        float phase = 0.10 + 0.90 * pow(max(dot(rd, uLightL), 0.0), 5.0);
        // uLightColor carries the day/night intensity, so the shafts fade out at night.
        // Raised from 0.00018 (#76): reference shader-pack screenshots show god rays as a
        // dominant compositional element through gaps in foliage/terrain, not a faint haze.
        light += uLightColor * (scatter * phase * len * 0.00032 * uShaftGain);
    }

    // Lamp light in the air. The cascades give the sun its shafts above; a lamp's light only
    // ever landed on faces, so a cave lit by one read as walls painted bright around a hole of
    // darkness, nothing between. This marches the voxel light around the camera (LightVolume,
    // filtered, so it is smooth between voxels where the faces are stepped) and adds what the
    // air along the ray would scatter back. Only what lamps and glow give: sky light in the air
    // is the distance haze's job.
    if (uVolOn > 0.5) {
        float len = min(dist, 56.0);
#if   VX_QUALITY == 0
        const int kLampSteps = 6;
#elif VX_QUALITY == 1
        const int kLampSteps = 8;
#elif VX_QUALITY == 3
        const int kLampSteps = 16;
#else
        const int kLampSteps = 12;
#endif
        float stepLen = len / float(kLampSteps);
        float t = stepLen * ign(gl_FragCoord.xy + vec2(37.0, 11.0));
        vec3 glow = vec3(0.0);
        for (int i = 0; i < kLampSteps; ++i) {
            vec3 p = rd * t;
            vec3 q = p - uVolMin;
            if (all(greaterThanEqual(q, vec3(0.0))) && all(lessThan(q, vec3(128.0)))) {
                vec3 l = texture(uLightVol, (uVolCam + p.xzy) / 128.0).rgb;
                // steeper than a lit face's l * l: the glow stays close round the light
                // instead of hazing the dark rock the ray passes on its way
                glow += l * l * l;
            }
            t += stepLen;
        }
        light += vec3(1.12, 1.0, 0.88) * glow * (stepLen * kLampAir);
    }
    outScatter = vec4(light, dist);
}
