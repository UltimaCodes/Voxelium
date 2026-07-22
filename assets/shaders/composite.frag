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
layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;

layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot; // view -> camera-relative world
layout(location = 3) uniform float uAoOn;
layout(location = 4) uniform float uVolumetricOn;
layout(location = 5) uniform vec3 uSunL;
layout(location = 6) uniform mat4 uCascadeVp[3];
layout(location = 18) uniform vec3 uCascadeSplit;

vec3 viewRay() {
    vec2 ndc = vUv * 2.0 - 1.0;
    return normalize(uInvViewRot * vec3(ndc.x * uProjScale.x, ndc.y * uProjScale.y, -1.0));
}

vec3 skyColor(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 zenith = vec3(0.14, 0.34, 0.80);
    vec3 horizon = vec3(0.66, 0.78, 0.92);
    vec3 col = mix(horizon, zenith, pow(up, 0.5));
    if (dir.y < 0.0) // below the horizon: fade to a hazy ground tone
        col = mix(horizon, vec3(0.34, 0.38, 0.44), clamp(-dir.y * 3.0, 0.0, 1.0));
    float sd = max(dot(dir, uSunL), 0.0);
    col += vec3(1.0, 0.55, 0.28) * pow(sd, 8.0) * 0.6;        // warm glow around the sun
    col += vec3(1.0, 0.95, 0.85) * smoothstep(0.9985, 0.9995, sd) * 9.0; // the disc
    return col;
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

    // sky pixel (reversed-z far = 0): paint the sky instead of the flat clear colour
    if (rawD <= 0.0) {
        outColor = vec4(skyColor(dir), 1.0);
        return;
    }

    vec3 color = texture(uScene, vUv).rgb;
    if (uAoOn > 0.5)
        color *= texture(uAo, vUv).r;

    if (uVolumetricOn > 0.5) {
        float viewZ = -uNear / max(rawD, 1e-7);
        vec2 ndc = vUv * 2.0 - 1.0;
        vec3 vpos = vec3(ndc.x * uProjScale.x * -viewZ, ndc.y * uProjScale.y * -viewZ, viewZ);
        vec3 world = uInvViewRot * vpos;
        float len = min(length(world), 150.0);
        vec3 rd = normalize(world);

        const int kSteps = 24;
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
        color += vec3(1.14, 1.05, 0.86) * (scatter * phase * len * 0.00032);
    }

    outColor = vec4(color, 1.0);
}
