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
flat in uint vWet; // the face looks out into water

layout(location = 0) out vec4 outColor; // HDR

// Toward whatever is lighting the world: the sun by day, the moon by night. The cascades are
// rendered from this same direction, so NdotL and the shadow lookup stay consistent after
// dusk. It is NOT "where the sun is" any more, use uDaylight for that.
layout(location = 4) uniform vec3 uSunL;
layout(location = 5) uniform float uShadowsOn;
layout(location = 6) uniform mat4 uCascadeVp[3]; // consumes 6..17
layout(location = 18) uniform vec3 uCascadeSplit; // radial split distances
layout(location = 19) uniform vec3 uCascadeTexel; // world units per shadow texel
layout(location = 21) uniform vec3 uLandSun;      // direct light on land, sun or moon
layout(location = 22) uniform vec3 uLandAmb;      // sky fill on land
layout(location = 23) uniform float uEmissive;    // emitter self-illumination, 0 = off
layout(location = 24) uniform vec3 uCamPos;       // camera world position
layout(location = 25) uniform float uCloudShadow; // cloud shadow strength, 0 = off
layout(location = 26) uniform float uFrame;       // moves the shadow noise for TAA, 0 without it
layout(location = 27) uniform float uLeafSss;    // leaf translucency strength, 0 = off
layout(location = 28) uniform float uRelief;     // derived surface relief, 0 = off
layout(location = 29) uniform float uDaylight;   // 1 full day, 0 after dusk
layout(location = 30) uniform float uWetness;    // 0 dry .. 1 soaked (#29)
layout(location = 34) uniform float uSnowCover;  // how much of the sky-facing ground has snow on it (33 is uCloudBias)
layout(location = 31) uniform float uCloudTime;  // the clouds' clock, for their shadows

layout(binding = 0) uniform sampler2DArray uBlockTex; // sRGB storage: reads back linear
layout(binding = 7) uniform sampler2D uClimate;  // moisture, temperature (FarTerrain)
layout(location = 35) uniform vec3 uClimateXf;   // x, z of its corner, 1 / span; z = 0 for none
layout(location = 36) uniform float uWashTime;   // seconds, wrapped: the wash on the beach
layout(binding = 9) uniform sampler2D uCoast;    // blocks inland, see FarTerrain
layout(location = 38) uniform vec3 uCoastXf;     // its corner relative to the camera, 1 / span; z = 0 for none
layout(location = 37) uniform float uBreakClock; // breakers come ashore one a unit of this
layout(location = 42) uniform vec4 uGroup;       // zw: the shoreline's wander, phased at the camera
layout(binding = 10) uniform sampler3D uLightVol; // voxel light around the camera (LightVolume), axes x, z, y
layout(location = 39) uniform vec3 uVolCam;      // camera in its texels
layout(location = 40) uniform vec3 uVolMin;      // its low corner relative to the camera
layout(location = 41) uniform float uVolOn;
layout(location = 43) uniform vec3 uHandLight;   // a lamp in the hand: its light levels, 0..15 a channel
layout(binding = 11) uniform sampler3D uBounce;  // sunlight off the ground nearby, LightVolume::computeBounce
layout(location = 44) uniform float uBounceOn;
#include "clouds.glsl"
#define CLOUD_WEATHER_BINDING 6
#include "cloudfield.glsl"
#include "texlayer.glsl"
#include "lighting.glsl"
layout(binding = 12) uniform sampler2D uCaustics; // the caustic web, a 64-block tile (caustics_tex.comp)

// What a block of water takes from each colour: red goes within a few blocks, blue lasts.
const vec3 kWaterSigma = vec3(0.22, 0.075, 0.03);
layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;
layout(binding = 5) uniform sampler2DArray uShadowDepth; // raw depth, for the PCSS blocker search

// 8 entries, not 6: dir 6 and 7 are the cross-plane foliage quads. They face up, so a
// tuft takes the sun the way the ground it grows out of does rather than going dark
// whenever you view it side-on. Indexing this with vDir at 6 would otherwise be out of
// bounds, which is undefined rather than merely wrong.
const vec3 kNormal[8] = vec3[8](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1),
                                vec3(0, 1, 0), vec3(0, 1, 0));
// Softened vanilla face shade: X sides, top, bottom, Z sides, then the two cross planes.
const float kFace[8] = float[8](0.72, 0.72, 1.0, 0.5, 0.84, 0.84, 1.0, 1.0);

// Tangent frames for the six axis-aligned faces, matching the (u,v) the vertex stage builds.
// Axis-aligned geometry means these are constants rather than anything that has to be
// stored per vertex, which matters, because the quad has no room left for tangent data.
const vec3 kTangent[8] = vec3[8](vec3(0, 0, 1), vec3(0, 0, -1), vec3(1, 0, 0),
                                 vec3(1, 0, 0), vec3(1, 0, 0), vec3(-1, 0, 0),
                                 vec3(1, 0, 0), vec3(1, 0, 0));
const vec3 kBitangent[8] = vec3[8](vec3(0, -1, 0), vec3(0, -1, 0), vec3(0, 0, 1),
                                   vec3(0, 0, 1), vec3(0, -1, 0), vec3(0, -1, 0),
                                   vec3(0, 0, 1), vec3(0, 0, 1));

const vec2 kPoisson[16] = vec2[16](
    vec2(-0.942, -0.399), vec2(0.946, -0.769), vec2(-0.094, -0.929), vec2(0.345, 0.294),
    vec2(-0.916, 0.458), vec2(-0.815, -0.879), vec2(-0.383, 0.277), vec2(0.975, 0.756),
    vec2(0.443, -0.975), vec2(0.537, -0.474), vec2(-0.265, -0.419), vec2(0.792, 0.191),
    vec2(-0.242, 0.997), vec2(-0.814, 0.914), vec2(0.200, 0.786), vec2(0.144, -0.141));

// Far cascade texels are already blocks wide: no blocker search there.
#if VX_QUALITY <= 1
const int kBlockerTaps[3] = int[3](5, 3, 0);
const int kPcfTaps[3] = int[3](8, 5, 3);
#else
const int kBlockerTaps[3] = int[3](8, 5, 0);
const int kPcfTaps[3] = int[3](12, 8, 4);
#endif

const float kPenumbra = 0.016; // blocks of blur per block between blocker and receiver

float screenNoise() {
    vec2 p = gl_FragCoord.xy + 5.588238 * uFrame;
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Everything here is sized in blocks, so softness and bias mean the same in every cascade.
float sampleCascade(vec3 pos, vec3 N, int c, float noise) {
    float slope = min(1.0 / max(max(dot(N, uSunL), 0.0), 0.2), 2.5);
    float texel = uCascadeTexel[c];
    float span = texel * VX_SHADOW_RES;
    float range = span + 160.0; // depth range of the cascade, see ShadowMap::cascadeMatrix

    vec4 sc = uCascadeVp[c] * vec4(pos + N * texel * 1.5 * slope, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float depth = sc.z - (0.35 + 0.6 * texel) * slope / range;

    float a = noise * 6.2831853;
    mat2 rot = mat2(cos(a), sin(a), -sin(a), cos(a));

    float blur = texel * 0.9;
    if (kBlockerTaps[c] > 0) {
        float search = max(3.0 * texel, 0.6) / span;
        float sum = 0.0;
        int hits = 0;
        for (int i = 0; i < kBlockerTaps[c]; ++i) {
            float d = texture(uShadowDepth, vec3(uv + rot * kPoisson[i] * search, float(c))).r;
            if (d < depth) {
                sum += d;
                ++hits;
            }
        }
        if (hits == 0)
            return 1.0;
        blur = clamp((depth - sum / float(hits)) * range * kPenumbra, 0.75 * texel, 1.5);
    }
    blur /= span;

    float lit = 0.0;
    for (int i = 0; i < kPcfTaps[c]; ++i)
        lit += texture(uShadowMap, vec4(uv + rot * kPoisson[i] * blur, float(c), depth));
    return lit / float(kPcfTaps[c]);
}

float sunShadow(vec3 pos, vec3 N) {
    if (uShadowsOn < 0.5)
        return 1.0;
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    float noise = screenNoise();
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    // near a split some pixels take the next cascade and TAA blends them, one lookup instead of two
    if (c < 2 && fract(noise * 7.0) < (d - uCascadeSplit[c] * 0.9) / (uCascadeSplit[c] * 0.1))
        ++c;
    return mix(sampleCascade(pos, N, c, noise), 1.0, smoothstep(uCascadeSplit.z * 0.9, uCascadeSplit.z, d));
}

// ---- natural variation -------------------------------------------------------------------
// A block texture is one 16x16 tile stamped on every block, and a beach or a hillside of it is
// visibly a repeating grid. Two cheap fixes, both keyed to the block's WORLD cell so they never
// change with view or LOD: each block gets one of eight orientations (four turns, mirrored or
// not), and its brightness is nudged a few percent, with a slow world-space drift on top so
// whole patches run a little warmer or cooler. Only ground-like tiles take part: bark grain,
// brick courses and strata have a direction that a turn would break, and leaves are excluded
// so the holes still match the shadow pass, which cannot see any of this.
bool naturalLayer(uint l) {
    return l == 0u || l == 1u || l == 2u || l == 8u || l == 9u || l == 12u || l == 13u ||
           l == 14u || l == 47u || l == 49u || l == 100u || l == 101u || l == 103u || l == 105u ||
           (l >= 108u && l <= 113u) || (l >= 119u && l <= 121u);
}

uint cellHash(ivec3 c) {
    uint h = uint(c.x) * 0x8DA6B343u ^ uint(c.y) * 0xD8163841u ^ uint(c.z) * 0xCB1AB31Fu;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = float(cellHash(ivec3(i, 7)) & 0xFFFFu), b = float(cellHash(ivec3(i + vec2(1, 0), 7)) & 0xFFFFu);
    float c = float(cellHash(ivec3(i + vec2(0, 1), 7)) & 0xFFFFu), d = float(cellHash(ivec3(i + vec2(1, 1), 7)) & 0xFFFFu);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y) / 65535.0;
}

// the linear part of a cell's orientation, applied to an offset from the tile centre
vec2 turnUv(vec2 c, uint hsh) {
    if ((hsh & 4u) != 0u)
        c.x = -c.x;
    uint r = hsh & 3u;
    if (r == 1u) c = vec2(-c.y, c.x);
    else if (r == 2u) c = -c;
    else if (r == 3u) c = vec2(c.y, -c.x);
    return c;
}

void main() {
    vec3 N = kNormal[vDir];
    vec2 uv = vUv;
    vec2 duvdx = dFdx(vUv), duvdy = dFdy(vUv);
    float natTint = 1.0;
    vec3 natWarm = vec3(1.0);
    uint natHash = 0u;
    bool natural = vDir < 6u && naturalLayer(vLayer);
    if (natural) {
        vec3 wp = uCamPos + vWorldRel;
        natHash = cellHash(ivec3(floor(wp - N * 0.5)));
        vec2 base = floor(vUv);
        uv = base + turnUv(fract(vUv) - 0.5, natHash) + 0.5;
        duvdx = turnUv(duvdx, natHash);
        duvdy = turnUv(duvdy, natHash);
        natTint = 0.95 + 0.10 * float((natHash >> 3) & 255u) / 255.0;
        // slow patches, ~30 blocks across: a little warmer/drier in some, cooler in others
        float drift = vnoise(wp.xz * 0.035) - 0.5;
        natTint *= 1.0 + 0.10 * drift;
        natWarm = vec3(1.0 + 0.16 * drift, 1.0 + 0.02 * drift, 1.0 - 0.20 * drift);
    }
    vec4 texel = natural ? textureGrad(uBlockTex, vec3(uv, float(vLayer)), duvdx, duvdy)
                         : texture(uBlockTex, vec3(vUv, float(vLayer)));
    // Alpha cutout. Two cases, same mechanism: cross-plane foliage (dir 6/7) is mostly
    // empty tile, and LEAVES are a solid block riddled with gaps. Leaves are deliberately
    // not occluders in the mesher either, so the faces behind them are still there to be
    // seen through those gaps, that is what makes a canopy read as leaves rather than as a
    // green cube, and it is why light and shadow get through it.
    if ((vDir >= 6u || isCutoutLayer(vLayer)) && texel.a < 0.5)
        discard;
    vec3 albedo = texel.rgb * natTint * natWarm;
    // Snow settles on whatever faces the sky and only where the sky reaches it: a roof or a canopy
    // overhead catches it and the ground beneath stays bare. The edge is broken up with noise so
    // it drifts and patches instead of being a clean contour.
    if (uSnowCover > 0.001 && vDir < 6u) {
        vec3 sw = uCamPos + vWorldRel;
        float up = smoothstep(0.55, 0.9, N.y);
        float open = vLight.a * vLight.a;
        float drift = vnoise(sw.xz * 0.8 + sw.y * 0.3);
        float amt = smoothstep(0.30, 0.70, uSnowCover * up * open * 1.05 + (drift - 0.5) * 0.7 * up);
        albedo = mix(albedo, vec3(0.79, 0.85, 0.95), amt);
    }

    // Grass side faces (layer 3) get the grass colour blended over their top edge. Without
    // it a terraced hillside alternates green tops with brown risers every single block,
    // which from altitude reads as corduroy. vUv.y counts down from the top on all four side
    // faces and tiles once per block, so 1 - fract() puts the overlay on each block's top edge, and a
    // greedy quad only ever merges grass sideways, never into a vertical run.
    float greenness = (vLayer == 2u || vLayer == 7u || isGrassLayer(vLayer)) ? 1.0 : (vLayer == 74u ? 0.7 : 0.0);
    if (vLayer == 3u) {
        const float kOverlay = 0.5; // top half of the block
        float g = smoothstep(1.0 - kOverlay, 1.0 - kOverlay * 0.45, 1.0 - fract(vUv.y));
        // the tile's own fringe counts too, so measured off how green the result is
        albedo = mix(albedo, texture(uBlockTex, vec3(vUv, 2.0)).rgb, g);
        greenness = smoothstep(0.02, 0.08, albedo.g - max(albedo.r, albedo.b));
    }
    if (greenness > 0.0 && uClimateXf.z > 0.0) {
        vec2 climate = texture(uClimate, (uCamPos.xz + vWorldRel.xz - uClimateXf.xy) * uClimateXf.z).rg;
        albedo = mix(albedo, albedo * climateTint(climate.r, climate.g), greenness);
    }
    // Beach at the waterline. Only a picture, the water itself stays put: a film running up the sand
    // after each breaker, foam along its front, wet sand where it has been.
    vec3 washSheen = vec3(0.0);
    if (vLayer == 47u && vDir == 2u && uCoastXf.z > 0.0) {
        float above = uCamPos.y + vWorldRel.y - 62.9;
        if (above < 2.6) {
            vec3 sw = uCamPos + vWorldRel;
            vec2 rel = vWorldRel.xz;
            float inland = textureLod(uCoast, (rel - uCoastXf.xy) * uCoastXf.z, 0.0).r;
            float wander = 0.6 * sin(dot(vec2(0.031, 0.017), rel) + uGroup.z) + 0.4 * sin(dot(vec2(-0.023, 0.041), rel) + uGroup.w);
            // a breaker reaches the shore at clock + wander * 0.26 = a whole number, the run up peaks after
            float run = 0.5 + 0.5 * cos(6.2831853 * (uBreakClock + wander * 0.26 - 0.22));
            float front = 2.0 + run * run * 11.0;
            float lace = vnoise(sw.xz * 0.31 + uWashTime * 0.09) * 0.6 + vnoise(sw.xz * 0.83 - uWashTime * 0.13) * 0.4;
            float reach = inland + (lace - 0.5) * 5.0;
            float film = smoothstep(front + 1.0, front - 1.5, reach);
            float edge = film * smoothstep(front - 3.2, front - 0.6, reach) * smoothstep(0.30, 0.55, lace);
            float low = 1.0 - smoothstep(1.2, 2.6, above);
            albedo = mix(albedo, albedo * vec3(0.58, 0.55, 0.52), max(film, smoothstep(17.0, 6.0, inland)));
            albedo = mix(albedo, albedo * vec3(0.74, 0.84, 0.90), film * low);
            albedo = mix(albedo, vec3(0.78, 0.82, 0.84), edge * low);
            washSheen = uLandAmb * (0.16 * film * low * (1.0 - edge));
        }
    }

    // Relief from the albedo's own luminance, on the lighting normal only and only close up.
    vec3 Nl = N;
    float reliefFade = 1.0 - smoothstep(14.0, 28.0, length(vWorldRel));
    if (uRelief > 0.0 && vDir < 6u && reliefFade > 0.01) {
        vec2 ts = 1.0 / vec2(textureSize(uBlockTex, 0).xy);
        const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
        // Stay inside this block's cell: past it the tile wraps and reads a made-up gradient.
        vec2 cellMin = floor(uv);
        vec2 cellMax = cellMin + 1.0;
        // One-sided differences off the texel already fetched above: two extra fetches
        // instead of four. Stepping toward the far side of the cell (and flipping the sign)
        // when the near side would leave it keeps the edge texels from falling back to zero.
        vec2 dir = vec2(uv.x + ts.x > cellMax.x ? -1.0 : 1.0, uv.y + ts.y > cellMax.y ? -1.0 : 1.0);
        float h0 = dot(texel.rgb, kLuma);
        float hx = dot(texture(uBlockTex, vec3(uv.x + dir.x * ts.x, uv.y, float(vLayer))).rgb, kLuma);
        float hy = dot(texture(uBlockTex, vec3(uv.x, uv.y + dir.y * ts.y, float(vLayer))).rgb, kLuma);
        // Scaled to match the two-texel central difference this replaced, and clamped so a
        // high-contrast tile cannot flip the normal through the surface.
        vec2 slope = clamp(vec2((h0 - hx) * dir.x, (h0 - hy) * dir.y) * 2.0 * uRelief * 2.5 * reliefFade,
                           -0.9, 0.9);
        if (natural) {
            // the slope was measured in the turned tile: T^T takes it back to the face's own frame
            vec2 ex = turnUv(vec2(1, 0), natHash), ey = turnUv(vec2(0, 1), natHash);
            slope = vec2(dot(slope, ex), dot(slope, ey));
        }
        Nl = normalize(N + kTangent[vDir] * slope.x + kBitangent[vDir] * slope.y);
    }

    // Near the camera light comes from the voxels themselves, per pixel and filtered, so it runs
    // smoothly across a face and round a corner. Past that, the quad's own.
    vec4 lightNow = vLight;
    float smoothLight = 0.0;
    if (uVolOn > 0.5) {
        vec3 volP = vWorldRel + N * 0.5;
        vec3 volQ = volP - uVolMin;
        float volEdge = min(min(min(volQ.x, volQ.y), volQ.z), min(min(128.0 - volQ.x, 128.0 - volQ.y), 128.0 - volQ.z));
        if (volEdge > 0.0) {
            smoothLight = smoothstep(0.0, 16.0, volEdge);
            // Rock reads as dark in the filter, which would black out every corner. It never falls
            // far below what the quad carries.
            vec4 v = max(texture(uLightVol, (uVolCam + volP.xzy) / 128.0), vLight * 0.62);
            lightNow = mix(vLight, v, smoothLight);
        }
    }
    // Light is 16 levels per block: outside the volume, dither so a falloff reads as a gradient.
    float dither = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) * (1.0 - smoothLight);
    vec3 blockRGB = clamp(lightNow.rgb + dither * (1.0 / 15.0), 0.0, 1.0);
    float skyL = clamp(lightNow.a + dither * (1.0 / 15.0), 0.0, 1.0);

    float emitMask = (uEmissive > 0.0 && isEmitterLayer(vLayer)) ? 1.0 : 0.0;
    bool foliage = vDir >= 6u || isCutoutLayer(vLayer);
    // A face under water is lit through it: how deep it is, from the sea level or, in a lake above it
    // or a cave, from how far the sky light has fallen (a level a block through water). The sky light
    // is spent on depth down there, so it no longer says whether the sun gets in; the cascades do.
    bool underWater = vWet != 0u;
    float depthW = underWater ? max(62.0 - (uCamPos.y + vWorldRel.y), (1.0 - lightNow.a) * 15.0) : 0.0;
    vec3 waterDown = exp(-kWaterSigma * depthW);
    float gate = underWater ? 1.0 : sunGate(skyL);
    float ndl = max(dot(Nl, uSunL), 0.0);
    float shadow = (gate > 0.0 && (ndl > 0.0 || foliage)) ? sunShadow(vWorldRel, N) : 0.0;
    float sunFacing = ndl;
    if (foliage) {
        // Thin and many-sided: lit from any angle on the sun side, and a leaf in the way lets some through.
        sunFacing = mix(ndl, 1.0, 0.4);
        shadow = mix(shadow, 1.0, 0.18);
    }
    float cloud = cloudBlock(uCamPos + vWorldRel, uSunL, uCloudShadow);

    vec3 sun = uLandSun * (sunFacing * mix(shadow, 1.0, emitMask) * gate * (1.0 - cloud));
    // what a cloud takes from the sun it scatters into the sky
    vec3 amb = skyFill(N, uLandAmb) * (1.0 + 0.2 * cloud) * skyLevel(skyL);
    vec3 fill = sunFill(N, uSunL, uLandSun) * skyLevel(skyL);
    if (underWater) {
        // the sun's slanting path down through the water, and the waves focusing it into caustics,
        // brightest on what faces up and fading as the light spreads with depth
        vec3 wp = uCamPos + vWorldRel;
        vec2 entry = wp.xz + uSunL.xz / max(uSunL.y, 0.25) * depthW; // where its light came through the surface
        float caus = texture(uCaustics, entry / 64.0).r * exp(-depthW * 0.16);
        sun *= exp(-kWaterSigma * depthW / max(uSunL.y, 0.25)) * (0.75 + 2.6 * caus * mix(0.3, 1.0, max(N.y, 0.0)));
        // A cave under water has no sky above at all, open sea that deep has little; both read 0.
        float open = mix(0.25, 1.0, smoothstep(0.0, 0.15, lightNow.a));
        amb = skyFill(N, uLandAmb) * waterDown * open;
        fill = sunFill(N, uSunL, uLandSun) * waterDown * (0.6 * open);
    }
    // Bounce: near the camera the flat fill gives way to the light the sunlit ground around actually
    // throws back, read a little out from the face so it leans toward where that ground is. Shade
    // beside a sunlit clearing or at a cave mouth warms up, shade with nothing lit near it goes
    // darker. Kept to where some sky gets in, the field does not stop at walls.
    if (uBounceOn > 0.5 && !underWater) {
        vec3 bp = vWorldRel + N * 1.5;
        vec3 bq = bp - uVolMin;
        float edge = min(min(min(bq.x, bq.y), bq.z), min(min(128.0 - bq.x, 128.0 - bq.y), 128.0 - bq.z));
        float w = smoothstep(0.0, 16.0, edge);
        if (w > 0.0) {
            float b = texture(uBounce, (uVolCam + bp.xzy) / 128.0).r;
            // it comes up off the ground: an overhang's underside gets the most, a wall half, a floor little
            float facing = 0.16 * (0.65 - 0.35 * N.y);
            vec3 thrown = uLandSun * vec3(1.0, 0.93, 0.82) * (facing * b) * smoothstep(0.0, 0.3, skyL) *
                          (1.0 - cloud);
            fill = mix(fill, fill * 0.55 + thrown, w);
        }
    }
    amb += fill;

    // The mesher bakes 0.68 + 0.32 * ao / 3 times the corner's share of the quad's light. Above 0.6
    // it is corner AO, below that a falloff into the neighbouring quad.
    float ao = vShade > 0.6 ? mix(0.55, 1.0, (vShade - 0.6) / 0.4) : mix(mix(0.15, 0.55, smoothLight), 0.55, vShade / 0.6);
    ao = mix(ao, 1.0, emitMask);
    vec3 scene = sun * mix(1.0, ao, 0.35) + amb * ao;

    vec3 torch = vec3(1.12, 1.0, 0.88) * (blockRGB * blockRGB * 2.1) * mix(mix(vShade, ao, smoothLight), 1.0, emitMask);
    // A lamp in the hand spreads like one placed, from just under the eye. Whatever the eye can see
    // it can reach, so it needs no shadows, and only the corners darken it (vShade also carries the
    // placed lights' falloff, which is nothing where none reach).
    if (uHandLight.r + uHandLight.g + uHandLight.b > 0.0) {
        // softer than a placed one: it is always a block or two from the nearest wall. Falls off by
        // its brightest channel and keeps its tint all the way out (main.cpp tones the tint down).
        float level = max(uHandLight.r, max(uHandLight.g, uHandLight.b));
        float hand = max(level - length(vWorldRel + vec3(0.0, 0.3, 0.0)), 0.0) / 15.0;
        torch = max(torch, vec3(1.12, 1.0, 0.88) * (uHandLight / level) * (hand * hand * 1.2) * max(ao, 0.55));
    }
    vec3 floorLight = mix(vec3(0.006, 0.0068, 0.0095), vec3(0.03, 0.034, 0.042), smoothstep(0.0, 0.35, skyL));

    // Backlit leaves and grass glow, gold against a low sun.
    vec3 transmit = vec3(0.0);
    if (uLeafSss > 0.0 && foliage) {
        float back = pow(max(dot(-uSunL, normalize(vWorldRel)), 0.0), 2.0);
        float through = max(dot(N, -uSunL), 0.0) * 0.35 + 0.65 * back;
        transmit = uLandSun * albedo * (1.5 * through * back * gate * shadow * uLeafSss);
    }

    // Wet: darker albedo and a sheen, only where rain reaches. Two lobes, one alone reads as plastic.
    float wet = uWetness * skyL * skyL * (1.0 - emitMask);
    vec3 wetSpec = vec3(0.0);
    if (wet > 0.001) {
        albedo *= mix(1.0, 0.7, wet);
        vec3 H = normalize(uSunL + normalize(-vWorldRel));
        float ndh = max(dot(N, H), 0.0);
        float gloss = foliage ? 0.0 : pow(ndh, 90.0) * 0.6;
        wetSpec = uLandSun * ((gloss + pow(ndh, 12.0) * 0.10) * wet * shadow * gate);
    }

    float face = foliage ? mix(kFace[vDir], 1.0, 0.5) : kFace[vDir];
    // A distant riser is a pixel or less, and treads and risers alternating there is not shape, it is
    // contour lines over every hill. They ease toward the tread's light.
    float farOff = smoothstep(180.0, 520.0, length(vWorldRel));
    if (farOff > 0.0 && vDir < 6u && N.y == 0.0)
        scene = mix(scene, uLandSun * uSunL.y * gate * (1.0 - cloud) + amb, farOff * 0.7);
    face = mix(face, 1.0, farOff * 0.6);
    vec3 emission = albedo * (uEmissive * 1.6 * emitMask);
    outColor = vec4(albedo * face * addLight(scene, torch, floorLight) + emission + wetSpec + transmit + washSheen * skyLevel(skyL), 1.0);
}
