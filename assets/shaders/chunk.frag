#version 460 core

in vec3 vWorldRel;
in vec2 vUv;
flat in uint vLayer;
flat in uint vDir;
flat in vec4 vLight;
in float vAO; // interpolated per-corner ambient occlusion

layout(location = 0) out vec4 outColor; // HDR

layout(location = 4) uniform vec3 uSunL;      // direction toward the sun
layout(location = 5) uniform float uShadowsOn;
layout(location = 6) uniform mat4 uCascadeVp[3]; // consumes 6..17
layout(location = 18) uniform vec3 uCascadeSplit; // radial split distances
layout(location = 19) uniform vec3 uCascadeTexel; // world units per shadow texel
layout(location = 21) uniform vec3 uSunColor;     // day/night direct radiance
layout(location = 22) uniform vec3 uSkyAmbient;   // day/night sky fill

layout(binding = 0) uniform sampler2DArray uBlockTex;
layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;
layout(binding = 5) uniform sampler2DArray uShadowDepth; // raw depth, for the PCSS blocker search

const vec3 kNormal[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));
// Per-face shading. The spread used to be 0.50..1.00, which reads fine on a wall but turns
// terraced ground into corduroy: looking down a slope you see a lit top, then the side of
// the block above, alternating every block, and a 2x brightness step between them makes
// every stair a hard band. Compressed to 0.66..1.00 — still clearly directional, no stripes.
const float kShade[6] = float[6](0.86, 0.86, 1.00, 0.66, 0.76, 0.76);

// fixed 16-tap Poisson disk — a soft penumbra without per-pixel rotation, so shadow
// edges don't crawl/shimmer when the camera moves (rotated PCF looks softer but needs
// TAA to be stable; fixed taps stay put).
const vec2 kPoisson[16] = vec2[16](
    vec2(-0.942, -0.399), vec2(0.946, -0.769), vec2(-0.094, -0.929), vec2(0.345, 0.294),
    vec2(-0.916, 0.458), vec2(-0.815, -0.879), vec2(-0.383, 0.277), vec2(0.975, 0.756),
    vec2(0.443, -0.975), vec2(0.537, -0.474), vec2(-0.265, -0.419), vec2(0.792, 0.191),
    vec2(-0.242, 0.997), vec2(-0.814, 0.914), vec2(0.200, 0.786), vec2(0.144, -0.141));

// PCSS: search for blockers to size the penumbra (sharp at contact, soft when the caster
// is far above the receiver), then PCF with that radius. shadow map is conventional depth
// (near light = smaller), so a blocker is a stored depth in front of the receiver.
float sampleCascade(vec3 pos, vec3 N, int c) {
    // normal-offset bias scaled by the cascade texel kills acne without peter panning
    vec4 sc = uCascadeVp[c] * vec4(pos + N * uCascadeTexel[c] * 1.5, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float receiver = sc.z;

    float searchR = 3.0 / 1536.0;
    float blockerSum = 0.0;
    int blockerN = 0;
    for (int i = 0; i < 16; ++i) {
        float d = texture(uShadowDepth, vec3(uv + kPoisson[i] * searchR, float(c))).r;
        if (d < receiver - 0.0016) {
            blockerSum += d;
            ++blockerN;
        }
    }
    if (blockerN == 0)
        return 1.0; // nothing between us and the sun

    float avgBlocker = blockerSum / float(blockerN);
    float penumbra = (receiver - avgBlocker) / max(avgBlocker, 1e-4);
    float radius = clamp(penumbra * 45.0, 1.0, 7.0) / 1536.0;

    float depth = receiver - 0.0016;
    float sum = 0.0;
    for (int i = 0; i < 16; ++i)
        sum += texture(uShadowMap, vec4(uv + kPoisson[i] * radius, float(c), depth));
    return sum / 16.0;
}

float sunShadow(vec3 pos, vec3 N) {
    if (uShadowsOn < 0.5)
        return 1.0;
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    float sh = sampleCascade(pos, N, c);
    // cross-fade into the next cascade near the split so there's no lighting seam
    float splitEnd = c == 0 ? uCascadeSplit.x : (c == 1 ? uCascadeSplit.y : uCascadeSplit.z);
    float blendStart = splitEnd * 0.82;
    if (c < 2 && d > blendStart)
        sh = mix(sh, sampleCascade(pos, N, c + 1), (d - blendStart) / (splitEnd - blendStart));
    // and fade shadows out at the very edge of the last cascade instead of a hard cut
    sh = mix(sh, 1.0, smoothstep(uCascadeSplit.z * 0.9, uCascadeSplit.z, d));
    return sh;
}

void main() {
    vec3 N = kNormal[vDir];
    vec3 albedo = texture(uBlockTex, vec3(vUv, float(vLayer))).rgb;

    // Grass side faces (layer 3) get the grass colour blended over their top edge. Without
    // it a terraced hillside alternates green tops with brown risers every single block,
    // which from altitude reads as corduroy. vUv.y is world Y on all four side faces and
    // tiles once per block, so fract() puts the overlay on each block's top edge — and a
    // greedy quad only ever merges grass sideways, never into a vertical run.
    if (vLayer == 3u) {
        const float kOverlay = 0.34; // top third of the block
        float g = smoothstep(1.0 - kOverlay, 1.0 - kOverlay * 0.45, fract(vUv.y));
        albedo = mix(albedo, texture(uBlockTex, vec3(vUv, 2.0)).rgb, g);
    }
    vec3 blockRGB = vLight.rgb; // per-channel so a coloured emitter tints its own light
    float skyL = vLight.a;

    float ndl = max(dot(N, uSunL), 0.0);
    float shadow = ndl > 0.0 ? sunShadow(vWorldRel, N) : 0.0;
    float skyAccess = skyL * skyL; // voxel sky light gates how much sun reaches here

    // warm strong sun vs cool dim sky ambient — the contrast is what keeps the scene
    // from going milky once tonemapped. the small floor stops fully-enclosed faces
    // (interior leaf pockets, deep cave walls) from crushing to pure black.
    vec3 direct = uSunColor * (ndl * shadow * skyAccess);
    vec3 ambient = uSkyAmbient * (0.10 + 0.90 * skyL) + vec3(0.035, 0.04, 0.05);
    vec3 torch = vec3(1.35, 0.95, 0.55) * (blockRGB * blockRGB * 1.6);

    // per-vertex ambient occlusion darkens concave corners smoothly (occluded -> 68%).
    // every terrace step is a concave corner, so a deep AO stacks with the face shading
    // and contributes its own dark line to the banding — kept subtler for that reason
    float ao = 0.68 + 0.32 * vAO;
    outColor = vec4(albedo * kShade[vDir] * (direct + ambient + torch) * ao, 1.0);
}
