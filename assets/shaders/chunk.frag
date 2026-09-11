#version 460 core

// Injected by Shader.cpp from the active quality preset (see submitProgram).
//
// VX_QUALITY picks the sample and step counts below. They MUST stay compile-time constants:
// a uniform loop bound does not unroll, and a quality system whose top setting is slower
// than having no quality system at all is the wrong trade. Default 2 = High, the build these
// numbers were tuned in, so an un-injected compile matches the one before presets existed.
//
// VX_SHADOW_RES must agree with the texture the cascades were actually allocated at, which
// is why it comes through the same channel. A define rather than a uniform because every use
// is a divisor sizing a tap offset, and a compile-time divisor folds away.
#ifndef VX_QUALITY
#define VX_QUALITY 2
#endif
#ifndef VX_SHADOW_RES
#define VX_SHADOW_RES 1536.0
#endif

in vec3 vWorldRel;
in vec2 vUv;
flat in uint vLayer;
flat in uint vDir;
flat in vec4 vLight;
in float vShade; // interpolated per-corner brightness: AO x this corner's share of vLight

layout(location = 0) out vec4 outColor; // HDR

// Toward whatever is lighting the world: the sun by day, the moon by night. The cascades are
// rendered from this same direction, so NdotL and the shadow lookup stay consistent after
// dusk. It is NOT "where the sun is" any more — use uDaylight for that.
layout(location = 4) uniform vec3 uSunL;
layout(location = 5) uniform float uShadowsOn;
layout(location = 6) uniform mat4 uCascadeVp[3]; // consumes 6..17
layout(location = 18) uniform vec3 uCascadeSplit; // radial split distances
layout(location = 19) uniform vec3 uCascadeTexel; // world units per shadow texel
layout(location = 21) uniform vec3 uSunColor;     // day/night direct radiance
layout(location = 22) uniform vec3 uSkyAmbient;   // day/night sky fill
layout(location = 23) uniform float uEmissive;    // emitter self-illumination, 0 = off
layout(location = 24) uniform vec3 uCamPos;       // camera world position
layout(location = 25) uniform float uCloudShadow; // cloud shadow strength, 0 = off
layout(location = 26) uniform float uTimeSky;     // same clock the cloud field drifts on
layout(location = 27) uniform float uLeafSss;    // leaf translucency strength, 0 = off
layout(location = 28) uniform float uRelief;     // derived surface relief, 0 = off
layout(location = 29) uniform float uDaylight;   // 1 full day, 0 after dusk
layout(location = 30) uniform float uWetness;    // 0 dry .. 1 soaked (#29)

#include "clouds.glsl"

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

// Tangent frames for the six axis-aligned faces, matching the (u,v) the vertex stage builds.
// Axis-aligned geometry means these are constants rather than anything that has to be
// stored per vertex — which matters, because the quad has no room left for tangent data.
const vec3 kTangent[8] = vec3[8](vec3(0, 0, 1), vec3(0, 0, -1), vec3(1, 0, 0),
                                 vec3(1, 0, 0), vec3(1, 0, 0), vec3(-1, 0, 0),
                                 vec3(1, 0, 0), vec3(1, 0, 0));
const vec3 kBitangent[8] = vec3[8](vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 0, 1),
                                   vec3(0, 0, 1), vec3(0, 1, 0), vec3(0, 1, 0),
                                   vec3(0, 0, 1), vec3(0, 0, 1));

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
    // Both bias terms below fight the same problem from two angles, and both need to grow
    // as the surface turns edge-on to the light (ndl -> 0). A shadow texel's world-space
    // footprint stays roughly constant, but the RECEIVER's depth changes faster and faster
    // across that footprint as the face goes grazing, so a fixed offset/bias tuned for a
    // face pointed at the sun stops covering a steep slope under a low sun — that exact
    // combination is what showed up as fine, regular self-shadow striping distinct from the
    // cloud-shadow bands (owner report, tour stop 0 at VX_TOUR_TIME 0.28). N is always one
    // of the six axis-aligned face directions here, so this is exact, not interpolated.
    // Floored at 0.2 rather than letting it blow up toward ndl=0 — sunShadow() already
    // zeroes this whole term below that via its own ndl>0 gate on the caller side, so a
    // shadow this function still gets asked to sample is never more than moderately
    // grazing, and the floor only needs to hold the worst case sane.
    float ndl = max(dot(N, uSunL), 0.0);
    float slope = 1.0 / max(ndl, 0.2);

    // normal-offset bias, in cascade-texel units so it means the same thing at every
    // distance, now ALSO scaled by slope so a grazing face is pushed further along its own
    // normal. Clamped (3x, not unbounded) so this cannot peter-pan a shadow off a thin
    // object the way an unclamped slope scale would.
    vec4 sc = uCascadeVp[c] * vec4(pos + N * uCascadeTexel[c] * 1.5 * min(slope, 3.0), 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float receiver = sc.z;

    // depth-compare bias: fixed at 0.0016 this was tuned for a face pointed at the sun.
    // Scaled by the same slope, clamped tighter (2.5x) since this one trades directly
    // against contact-shadow precision rather than just nudging the sample position.
    float depthBias = 0.0016 * min(slope, 2.5);

    float searchR = 3.0 / VX_SHADOW_RES;
    float blockerSum = 0.0;
    int blockerN = 0;
#if   VX_QUALITY <= 1
    const int kBlockerTaps = 8;
#else
    const int kBlockerTaps = 16;
#endif
    for (int i = 0; i < kBlockerTaps; ++i) {
        float d = texture(uShadowDepth, vec3(uv + kPoisson[i] * searchR, float(c))).r;
        if (d < receiver - depthBias) {
            blockerSum += d;
            ++blockerN;
        }
    }
    if (blockerN == 0)
        return 1.0; // nothing between us and the sun

    float avgBlocker = blockerSum / float(blockerN);
    float penumbra = (receiver - avgBlocker) / max(avgBlocker, 1e-4);
    float radius = clamp(penumbra * 45.0, 1.0, 7.0) / VX_SHADOW_RES;

    float depth = receiver - depthBias;
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
    // Alpha cutout. Two cases, same mechanism: cross-plane foliage (dir 6/7) is mostly
    // empty tile, and LEAVES are a solid block riddled with gaps. Leaves are deliberately
    // not occluders in the mesher either, so the faces behind them are still there to be
    // seen through those gaps — that is what makes a canopy read as leaves rather than as a
    // green cube, and it is why light and shadow get through it.
    if ((vDir >= 6u || vLayer == 7u) && texel.a < 0.5)
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
    // Surface relief WITHOUT an authored normal map. #38 was blocked on texture authoring
    // and on the quad being full (no room for tangent data), and both go away if the height
    // field is the albedo's own luminance: on pixel-art tiles the dark pixels ARE the mortar
    // lines, cracks and gaps, and the light ones are the raised faces. That is what a height
    // map encodes, so the gradient of it is the normal.
    //
    // Only the LIGHTING normal is perturbed. The geometric normal still drives the shadow
    // bias and the face-shade table, because tilting those produces acne and makes flat
    // faces read as different brightnesses.
    // Distance gated, and that is most of what makes it affordable. Relief is per-TEXEL
    // detail: past a couple of dozen blocks a whole tile is a few pixels and the four extra
    // samples buy nothing you can see. Interleaved A/B measured the ungated version at
    // +0.7 ms scene / +1.0 ms frame, and the overwhelming majority of fragments in any view
    // are far ones.
    vec3 Nl = N;
    float reliefFade = 1.0 - smoothstep(20.0, 34.0, length(vWorldRel));
    if (uRelief > 0.0 && vDir < 6u && reliefFade > 0.01) {
        vec2 ts = 1.0 / vec2(textureSize(uBlockTex, 0).xy);
        const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
        float hl = dot(texture(uBlockTex, vec3(vUv - vec2(ts.x, 0.0), float(vLayer))).rgb, kLuma);
        float hr = dot(texture(uBlockTex, vec3(vUv + vec2(ts.x, 0.0), float(vLayer))).rgb, kLuma);
        float hd = dot(texture(uBlockTex, vec3(vUv - vec2(0.0, ts.y), float(vLayer))).rgb, kLuma);
        float hu = dot(texture(uBlockTex, vec3(vUv + vec2(0.0, ts.y), float(vLayer))).rgb, kLuma);
        // Scaled by the texture size so the slope is resolution independent, and clamped so
        // a high-contrast tile cannot flip the normal through the surface.
        vec2 slope = clamp(vec2(hl - hr, hd - hu) * uRelief * 2.5 * reliefFade, -0.9, 0.9);
        Nl = normalize(N + kTangent[vDir] * slope.x + kBitangent[vDir] * slope.y);
    }

    vec3 blockRGB = vLight.rgb; // per-channel so a coloured emitter tints its own light
    float skyL = vLight.a;

    float ndl = max(dot(Nl, uSunL), 0.0);
    float shadow = ndl > 0.0 ? sunShadow(vWorldRel, N) : 0.0;
    float skyAccess = skyL * skyL; // voxel sky light gates how much sun reaches here

    // warm strong sun vs cool dim sky ambient — the contrast is what keeps the scene
    // from going milky once tonemapped. the small floor stops fully-enclosed faces
    // (interior leaf pockets, deep cave walls) from crushing to pure black.
    // Cloud shadows sample the same field the sky raymarches, so a shadow always belongs
    // to a cloud you can actually see overhead. Only the direct sun term is affected —
    // ambient is sky light, which a cloud scatters rather than blocks.
    float clouded = uCloudShadow > 0.0
                        ? mix(1.0, cloudSunlight(uCamPos + vWorldRel, uSunL, uTimeSky),
                              uCloudShadow)
                        : 1.0;
    // Emitters: layer 4 is glow, 15..39 the 25 lamp swatches. Needed in two places below,
    // so it is worked out once here rather than written out twice and allowed to disagree.
    float emitMask =
        (uEmissive > 0.0 && (vLayer == 4u || (vLayer >= 15u && vLayer <= 39u))) ? 1.0 : 0.0;

    vec3 direct = uSunColor * (ndl * shadow * skyAccess * clouded);

    // How hard sky access gates the ambient fill. In daylight the sun dominates and a tight
    // gate (0.10 floor, a 10x range) is what gives caves and overhangs their depth.
    //
    // The floor used to rise to 0.45 after dark, because night had no directional light at
    // all and ambient WAS the entire image — every hollow went hard black with nothing to
    // fill it. The moon now carries that shape (see computeSky), so the fill no longer has
    // to, and 0.45 was flattening and brightening every night surface to buy something it
    // is no longer buying. 0.20 keeps night light more omnidirectional than the sun, which
    // is honest for scattered skylight, without erasing the relief.
    float gateFloor = mix(0.20, 0.10, uDaylight);
    // The flat term used to be an ungated constant, which reads as harmless next to a
    // daylight ambient of 0.26-0.55 but was 40-64% of the night ambient — a day-neutral
    // number that was, in practice, a night-only brightener. Faded out with daylight so it
    // still stops fully-enclosed faces (interior leaf pockets, deep cave walls) crushing to
    // pure black in daylight, without lifting the whole night off the floor.
    vec3 ambient = uSkyAmbient * (gateFloor + (1.0 - gateFloor) * skyL) +
                   vec3(0.035, 0.04, 0.05) * mix(0.25, 1.0, uDaylight);
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
    // Leaf subsurface scattering. A leaf is thin and translucent: with the sun behind it,
    // light comes THROUGH rather than bouncing off, which is why a canopy lit from behind
    // glows instead of silhouetting flat black. The wrap term is the standard cheap stand-in
    // for a real transmission lobe — a back-facing lobe around -L, tightened with a power so
    // it only fires when you are actually looking toward the sun through the leaf.
    //
    // Layers 7 (leaves) and 40/41 (the cross-plane tuft and flower) are the thin geometry;
    // everything else is opaque and gets nothing. Scaled by skyAccess so leaves buried deep
    // inside a canopy, which no sunlight reaches, do not light up.
    vec3 transmit = vec3(0.0);
    if (uLeafSss > 0.0 && (vLayer == 7u || vLayer == 40u || vLayer == 41u)) {
        float back = pow(max(dot(-uSunL, normalize(vWorldRel)), 0.0), 3.0);
        // wrap: some light gets through even side-on, which keeps it from popping in
        float wrap = max(dot(N, -uSunL), 0.0) * 0.35 + 0.65 * back;
        transmit = uSunColor * albedo * wrap * back * skyAccess * shadow * uLeafSss;
    }

    // Wet surfaces (#29). Three things happen to a surface in the rain, and all three are
    // needed or it just looks darker: it darkens (a water film traps light in the
    // microsurface), it gets a specular lobe (the film is smooth where the block is not),
    // and it does neither where the rain cannot reach.
    //
    // skyL is that last test, already sitting in the quad: it is the voxel sky access the
    // sun term uses, so a cave floor, an overhang and the underside of a ledge are dry for
    // free. That is the whole reason this hooks in here rather than in a screen-space pass,
    // which would have no idea what is under cover.
    //
    // Emitters stay dry-looking: a lamp's own light swamps a specular highlight anyway, and
    // darkening the albedo of a light source is the #49 mistake again.
    float wet = uWetness * skyL * skyL * (1.0 - emitMask);
    vec3 wetSpec = vec3(0.0);
    if (wet > 0.001) {
        albedo *= mix(1.0, 0.62, wet);              // damp, not repainted
        vec3 V = normalize(-vWorldRel);
        vec3 H = normalize(uSunL + V);
        // Two lobes for the same reason the water has two: one narrow highlight on its own
        // reads as plastic. The broad one is what makes a wet road look wet.
        float ndh = max(dot(N, H), 0.0);
        float spec = pow(ndh, 90.0) * 1.5 + pow(ndh, 12.0) * 0.16;
        // Shadowed by the cascade, so a wet surface under a tree does not glint.
        wetSpec = uSunColor * spec * wet * shadow * skyAccess;
    }

    vec3 emission = albedo * uEmissive * emitMask + wetSpec;

    // Light a surface RECEIVES is ambient-occluded. Light it PRODUCES is not: AO answers
    // "how much ambient reaches this corner", which means nothing for the face that is the
    // source. That distinction is the whole of #49 — an emitter's own torch term is by far
    // the largest thing in its shading, so AO on it puts a dark band along the bottom of
    // every lamp sitting on the ground, exactly where the corners are most occluded. It
    // reads as the block being shadowed at its base while it is lighting everything around
    // it. kShade stays, so a lamp still reads as a cube rather than a flat sticker.
    float ownShade = mix(vShade, 1.0, emitMask);
    // Ambient is gated TWICE, and that is what crushes a shaded terrace face to black.
    // Once by skyL above (the voxel sky access), and again by vShade — which is AO times
    // that corner's share of the quad's light, and the light includes sky. So sky access
    // is applied on top of itself, squared, on exactly the surfaces that have least of it:
    // the inside faces of a notch, the risers of a terrace, anything under an overhang.
    //
    // AO genuinely does belong on ambient; the corner's light share does not, because
    // skyL already answered that question. They are baked into one 3-bit value at mesh
    // time and cannot be separated here, so the corner term is softened for the ambient
    // path rather than removed: at worst it takes 45% instead of everything. `direct`
    // keeps the full term — a face the sun does not reach should still go dark.
    float ambShade = mix(0.55, 1.0, vShade);
    outColor = vec4(albedo * kShade[vDir] *
                            (direct * vShade + ambient * ambShade + torch * ownShade) +
                        emission + transmit,
                    1.0);
}
