#version 460 core

in vec3 vWorldRel;
in vec2 vUv;
flat in uint vLayer;
flat in uint vDir;
flat in vec4 vLight;
in float vShade; // interpolated per-corner brightness: AO x this corner's share of vLight

layout(location = 0) out vec4 outColor; // HDR

layout(location = 4) uniform vec3 uSunL;      // direction toward the sun
layout(location = 5) uniform float uShadowsOn;
layout(location = 6) uniform mat4 uCascadeVp[3]; // consumes 6..17
layout(location = 18) uniform vec3 uCascadeSplit; // radial split distances
layout(location = 19) uniform vec3 uCascadeTexel; // world units per shadow texel
layout(location = 21) uniform vec3 uSunColor;     // day/night direct radiance
layout(location = 22) uniform vec3 uSkyAmbient;   // day/night sky fill
layout(location = 23) uniform float uEmissive;    // emitter self-illumination, 0 = off

layout(binding = 0) uniform sampler2DArray uBlockTex;
layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;
layout(binding = 5) uniform sampler2DArray uShadowDepth; // raw depth, for the PCSS blocker search

// 8 entries, not 6: dir 6 and 7 are the cross-plane foliage quads. They face up, so a
// tuft takes the sun the way the ground it grows out of does rather than going dark
// whenever you view it side-on. Indexing this with vDir at 6 would otherwise be out of
// bounds, which is undefined rather than merely wrong.
const vec3 kNormal[8] = vec3[8](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1),
                                vec3(0, 1, 0), vec3(0, 1, 0));
// Per-face shading. The spread used to be 0.50..1.00, which reads fine on a wall but turns
// terraced ground into corduroy: looking down a slope you see a lit top, then the side of
// the block above, alternating every block, and a 2x brightness step between them makes
// every stair a hard band. Compressed to 0.66..1.00 — still clearly directional, no stripes.
const float kShade[8] = float[8](0.86, 0.86, 1.00, 0.66, 0.76, 0.76, 1.0, 1.0);

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
    vec4 texel = texture(uBlockTex, vec3(vUv, float(vLayer)));
    // Cross-plane foliage is alpha cut out: the tile is mostly empty and the empty part has
    // to leave the world behind it showing, not a transparent-black rectangle.
    if (vDir >= 6u && texel.a < 0.5)
        discard;
    vec3 albedo = texel.rgb;

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
    // The emitter's own colour drives the tint now (M18), so the fixed warm multiplier
    // that used to supply it would double-tint. Keep a slight warm bias for the ember
    // feel and let the light itself carry the hue out to its falloff.
    vec3 torch = vec3(1.12, 1.0, 0.88) * (blockRGB * blockRGB * 2.1);

    // Per-corner brightness, interpolated across the face. It already carries the old AO
    // term (mesher bakes 0.68 + 0.32 * ao into it, so concave corners darken exactly as
    // much as they used to) multiplied by how much of the quad's light this corner
    // actually sees — that second half is what ramps a merged quad into its neighbour
    // instead of stepping. A quad with uniform corners is all 7s and shades flat, so open
    // terrain looks and costs the same as before.
    // An emitter's own face was only ever LIT by its own light, which made a lamp read as a
    // brightly lit block rather than a light — and left it hovering around the bloom
    // threshold, so the glow came and went with the face angle. Emission is added flat:
    // not scaled by kShade, AO or vShade, because none of those mean anything for a surface
    // that is producing the light rather than receiving it.
    //
    // The hue comes from the albedo, which for a lamp IS its swatch colour (the tiles are
    // generated from blocks::kLampRGB), so this needs no second copy of that table to drift
    // out of sync. Layer 4 is glow, 15..39 are the 25 lamp swatches — see texLayer() in
    // chunk.vert, which owns that mapping.
    vec3 emission = vec3(0.0);
    if (uEmissive > 0.0 && (vLayer == 4u || (vLayer >= 15u && vLayer <= 39u)))
        emission = albedo * uEmissive;

    outColor = vec4(albedo * kShade[vDir] * (direct + ambient + torch) * vShade + emission,
                    1.0);
}
