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
// dusk. It is NOT "where the sun is" any more, use uDaylight for that.
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
layout(location = 34) uniform float uSnowCover;  // how much of the sky-facing ground has snow on it (33 is uCloudBias)
layout(location = 31) uniform vec3 uCloudMapXf;  // cloud map: world XZ of its corner, 1 / span

#include "clouds.glsl"
#include "texlayer.glsl"

layout(binding = 0) uniform sampler2DArray uBlockTex;
layout(binding = 6) uniform sampler2D uCloudMap; // coverage field, see cloudmap.frag
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
// every stair a hard band. Compressed to 0.66..1.00, still clearly directional, no stripes.
const float kShade[8] = float[8](0.86, 0.86, 1.00, 0.66, 0.76, 0.76, 1.0, 1.0);

// Tangent frames for the six axis-aligned faces, matching the (u,v) the vertex stage builds.
// Axis-aligned geometry means these are constants rather than anything that has to be
// stored per vertex, which matters, because the quad has no room left for tangent data.
const vec3 kTangent[8] = vec3[8](vec3(0, 0, 1), vec3(0, 0, -1), vec3(1, 0, 0),
                                 vec3(1, 0, 0), vec3(1, 0, 0), vec3(-1, 0, 0),
                                 vec3(1, 0, 0), vec3(1, 0, 0));
const vec3 kBitangent[8] = vec3[8](vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 0, 1),
                                   vec3(0, 0, 1), vec3(0, 1, 0), vec3(0, 1, 0),
                                   vec3(0, 0, 1), vec3(0, 0, 1));

// fixed 16-tap Poisson disk, a soft penumbra without per-pixel rotation, so shadow
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
    // face pointed at the sun stops covering a steep slope under a low sun, that exact
    // combination is what showed up as fine, regular self-shadow striping distinct from the
    // cloud-shadow bands (owner report, tour stop 0 at VX_TOUR_TIME 0.28). N is always one
    // of the six axis-aligned face directions here, so this is exact, not interpolated.
    // Floored at 0.2 rather than letting it blow up toward ndl=0, sunShadow() already
    // zeroes this whole term below that via its own ndl>0 gate on the caller side, so a
    // shadow this function still gets asked to sample is never more than moderately
    // grazing, and the floor only needs to hold the worst case sane.
    float ndl = max(dot(N, uSunL), 0.0);
    float slope = 1.0 / max(ndl, 0.2);

    // Owner report: hard black patches on terrain, water AND foliage, present only in the
    // day, i.e. it scales with the direct-light shadow term specifically, not ambient (that
    // was #58, a different bug, already fixed). Root cause: every one of kPoisson's 16 offsets
    // is IDENTICAL for every pixel on screen, both here and in the PCF loop below. Wherever
    // that fixed pattern straddles a voxel-scale edge in the shadow depth map, a whole patch
    // of neighbouring pixels gets the same wrong blocker-found/not-found answer together,
    // that reads as a hard, blocky dark shape, not the soft penumbra PCSS is supposed to
    // produce, and it is worst exactly where this session's tour capture found it: complex
    // close-range self-shadowing (terraces, foliage interiors). Rotating the disk per pixel
    // breaks the spatial correlation, neighbouring pixels now sample different actual
    // offsets for "the same" Poisson index, so the same misalignment cannot land on a whole
    // patch at once, only scattered individual texels. That turns it into fine dither instead
    // of a coherent shape, exactly the trick `ssgi.frag`'s kernel already uses (its own
    // comment: "the kernel rotates per frame so TAA integrates the noise into a smooth
    // result"), this engine already has TAA (`taa.frag`, ~10 frames of accumulation), which
    // is precisely the infrastructure that resolves per-pixel noise like this into something
    // smooth over a couple of frames of real gameplay, even though a single static capture
    // (as used to verify this) still shows it as fine dither rather than fully resolved.
    float rotA = fract(sin(dot(gl_FragCoord.xy, vec2(35.17, 91.73))) * 27321.58) * 6.2831853;
    mat2 rot = mat2(cos(rotA), sin(rotA), -sin(rotA), cos(rotA));

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
    // Taps per cascade. The far two cover texels that are already blocks wide, so extra taps
    // there resolve nothing the eye can see; the near one is where the penumbra is looked at.
    // Shadow sampling was ~40% of the scene pass at 16+16 taps everywhere.
#if   VX_QUALITY <= 1
    const int kBlk[3] = int[3](5, 3, 3);
    const int kPcf[3] = int[3](8, 5, 4);
#elif VX_QUALITY == 3
    const int kBlk[3] = int[3](16, 10, 6);
    const int kPcf[3] = int[3](16, 12, 8);
#else
    const int kBlk[3] = int[3](8, 5, 4);
    const int kPcf[3] = int[3](12, 8, 6);
#endif
    const int kBlockerTaps = kBlk[c];
    for (int i = 0; i < kBlockerTaps; ++i) {
        float d = texture(uShadowDepth, vec3(uv + rot * kPoisson[i] * searchR, float(c))).r;
        if (d < receiver - depthBias) {
            blockerSum += d;
            ++blockerN;
        }
    }
    if (blockerN == 0)
        return 1.0; // nothing between us and the sun

    float avgBlocker = blockerSum / float(blockerN);
    float penumbra = (receiver - avgBlocker) / max(avgBlocker, 1e-4);
    // Upper bound raised from 7.0 (#77): reference shadows read as very soft, almost diffuse
    // at distance from the caster, and 7 texels capped softness well short of that even for
    // a caster far above the receiver. Contact shadows are unaffected, a small penumbra
    // value still maps to a small radius, this only raises the ceiling for genuinely distant
    // casters.
    float radius = clamp(penumbra * 45.0, 1.0, 11.0) / VX_SHADOW_RES;

    float depth = receiver - depthBias;
    float sum = 0.0;
    const int kPcfTaps = kPcf[c];
    for (int i = 0; i < kPcfTaps; ++i)
        sum += texture(uShadowMap, vec4(uv + rot * kPoisson[i] * radius, float(c), depth));
    return sum / float(kPcfTaps);
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
    float blendStart = splitEnd * 0.9;
    if (c < 2 && d > blendStart)
        sh = mix(sh, sampleCascade(pos, N, c + 1), (d - blendStart) / (splitEnd - blendStart));
    // and fade shadows out at the very edge of the last cascade instead of a hard cut
    sh = mix(sh, 1.0, smoothstep(uCascadeSplit.z * 0.9, uCascadeSplit.z, d));
    return sh;
}

// How much sunlight gets through the cloud layer to worldPos, 1 = clear. The coverage field
// is rendered once a frame into uCloudMap (cloudmap.frag), so this is three filtered fetches
// along the sun ray instead of the per-fragment fbm it used to be.
const float kCloudMaxBlock = 0.5;

float cloudSunlight(vec3 worldPos, vec3 sunDir) {
    if (sunDir.y <= 0.05 || worldPos.y >= kCloudBase)
        return 1.0;
    // A run of terrace steps shares one lateral sample instead of jittering block to block:
    // 1/sunDir.y turns a single block of height into a big sideways jump at a low sun.
    float shadowY = round(worldPos.y / 24.0) * 24.0;
    float shade = 0.0;
    float weight = 0.0;
    float edge = 1.0;
    for (int i = 0; i < 3; ++i) {
        float f = 0.15 + 0.35 * float(i); // spread through the slab, away from its feathered ends
        float t = (mix(kCloudBase, kCloudTop, f) - shadowY) / sunDir.y;
        vec2 uv = (worldPos.xz + sunDir.xz * t - uCloudMapXf.xy) * uCloudMapXf.z;
        // the layer's own density profile as the tap weight, so its thin base and top do not
        // count as much as the solid middle
        float w = smoothstep(0.0, 0.22, f) * (1.0 - smoothstep(0.55, 1.0, f)) + 0.08;
        shade += w * textureLod(uCloudMap, uv, 1.5).r;
        weight += w;
        edge = min(edge, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
    }
    float lit = 1.0 - 0.6 * (shade / weight);
    // a very low sun throws the ray off the map; fade to clear instead of smearing the border
    return mix(1.0, lit, smoothstep(0.0, 0.08, edge));
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
           l == 14u || l == 47u || l == 49u;
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
        albedo = mix(albedo, vec3(0.90, 0.93, 0.98), amt);
    }

    // Grass side faces (layer 3) get the grass colour blended over their top edge. Without
    // it a terraced hillside alternates green tops with brown risers every single block,
    // which from altitude reads as corduroy. vUv.y is world Y on all four side faces and
    // tiles once per block, so fract() puts the overlay on each block's top edge, and a
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
    float reliefFade = 1.0 - smoothstep(14.0, 28.0, length(vWorldRel));
    if (uRelief > 0.0 && vDir < 6u && reliefFade > 0.01) {
        vec2 ts = 1.0 / vec2(textureSize(uBlockTex, 0).xy);
        const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
        // Owner report: a hard dark line at every block border on stepped/terraced ground,
        // with the interior of the same face reading normally lit, the opposite of what a
        // real per-step cast shadow would look like. Cause: vUv tiles once per BLOCK, and
        // uBlockTex is GL_REPEAT, so sampling past vUv's own [0,1] span here does not read
        // the actual neighbouring quad, each greedy-meshed quad is its own texture lookup
        // with no knowledge of what is drawn next to it, so a wrapped sample just re-reads
        // THIS quad's own opposite edge. On a large flat run of one uniform texture that is
        // invisible (every block repeats the same tile, so wrapping to "your own start" and
        // wrapping to "the real neighbour" look identical), which is exactly why this went
        // unnoticed generally. It becomes very visible on terraced/stepped ground, because
        // AO forces a fresh, UNMERGED quad at nearly every step (greedy meshing cannot merge
        // across differing per-corner AO), so almost every step edge IS a real quad boundary,
        // and the wrapped sample there reads a fabricated gradient with no relation to the
        // real neighbour's texture or shading, a fake dark line baked into the lighting
        // normal at exactly the border chunk.frag's own geometric AO already shades
        // correctly. Clamped to this quad's own UV span instead: at the very edge this falls
        // back to a one-sided difference (re-reading the centre texel) rather than a
        // fabricated wrap, which is far less wrong than reading an unrelated tile.
        // Clamped to THIS texel's own block cell, not a flat [0,1], a merged quad spanning
        // several blocks of the same material legitimately carries vUv past 1.0, tiling via
        // the same GL_REPEAT, and a flat clamp would wrongly flatten relief across all of it
        // past the first block.
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

    // Owner report: light visibly steps as it falls off, in hard bands rather than a smooth
    // gradient. Real cause is storage, not this shader, sky/block light is 4 bits per
    // channel (16 levels) in Chunk's packed word, and BFS propagation drops exactly 1 level
    // per block, so a falloff is a stack of 16 flat shells with a genuine value jump at each
    // boundary, not a quantized-from-continuous signal. Widening that storage (and re-tuning
    // the whole propagation/falloff to match) is a real, separate, much bigger change than a
    // shader edit, this is the standard, immediate mitigation for visible banding from
    // coarse source data: dither the value spatially before it is used, so a hard shell
    // boundary becomes a noisy transition band instead of a crisp line. It does not create
    // information the light data does not have, so it will not look truly continuous
    // up close, it makes the existing steps read as texture instead of a contour.
    float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    vec3 blockRGB = clamp(vLight.rgb + dither * (1.0 / 15.0), 0.0, 1.0);
    float skyL = clamp(vLight.a + dither * (1.0 / 15.0), 0.0, 1.0);

    // Far terrain is made of terraces a pixel or less deep, and a lit tread next to a shaded
    // riser then alternates every pixel: moire that crawls under TAA's jitter. From a distance
    // that alternation is not information, so it is faded toward the face-averaged look
    // (normal tipped toward up, riser darkening and per-corner AO eased off).
    float farK = smoothstep(80.0, 300.0, length(vWorldRel));
    Nl = normalize(mix(Nl, vec3(0.0, 1.0, 0.0), 0.55 * farK));
    float ndl = max(dot(Nl, uSunL), 0.0);
    float skyAccess = skyL * skyL; // voxel sky light gates how much sun reaches here
    // Nothing below ~1.4% reaches a sealed-in fragment anyway (direct is scaled by skyAccess),
    // so caves and canopy interiors skip the whole PCSS lookup.
    float shadow = (ndl > 0.0 && skyL > 0.12) ? sunShadow(vWorldRel, N) : 0.0;

    // warm strong sun vs cool dim sky ambient, the contrast is what keeps the scene
    // from going milky once tonemapped. the small floor stops fully-enclosed faces
    // (interior leaf pockets, deep cave walls) from crushing to pure black.
    // Cloud shadows sample the same field the sky raymarches, so a shadow always belongs
    // to a cloud you can actually see overhead. Only the direct sun term is affected,
    // ambient is sky light, which a cloud scatters rather than blocks.
    float clouded = uCloudShadow > 0.0
                        ? mix(1.0, cloudSunlight(uCamPos + vWorldRel, uSunL), uCloudShadow)
                        : 1.0;
    // A cloud is translucent: it takes at most about half the sun. Full strength made a cloud
    // shadow as dark as the shadow of a solid block, so the two could not be told apart.
    clouded = 1.0 - kCloudMaxBlock * (1.0 - clouded);
    // Emitters: layer 4 is glow, 15..39 the 25 lamp swatches. Needed in several places below,
    // so it is worked out once here rather than written out over and allowed to disagree.
    float emitMask = (uEmissive > 0.0 && isEmitterLayer(vLayer)) ? 1.0 : 0.0;

    // A light's own face does not take the cascade shadow. In daylight the sun is a large part
    // of what a lamp face shows, so a neighbour's shadow cut a hard dark edge straight across
    // the glow (INFO #12), which is not what a light in shade looks like. It keeps the sun's
    // angle, and sky access still takes the sun away entirely underground.
    vec3 direct = uSunColor * (ndl * mix(shadow, 1.0, emitMask) * skyAccess * clouded);
    // A cloud does not delete the light it blocks, it scatters it. Under one the ground gets
    // a wide, soft, shadowless source instead of a hole in the sun, which is why real cloud
    // shadows are gentle and why ours read as harsh dark patches when they only subtracted.
    vec3 cloudFill = uSunColor * ((1.0 - clouded) * 0.4 * skyAccess * mix(0.65, 1.0, ndl));

    // How hard sky access gates the ambient fill. Was a tight 0.10 daytime floor, raised to
    // 0.4 in response to repeated "hard black" reports, but that turned out to be
    // misattributed: the actual causes were contact shadows at 3x their sane strength,
    // cloud shadow's comb-filter aliasing, and SSAO reading through water, all separately
    // root-caused and fixed since. With those gone, 0.4 is no longer buying real bug
    // avoidance, only flattening legitimate shadow depth, and stacked with this session's
    // SSGI increase (both inject fill light into the same shadowed areas), the owner's own
    // report is exactly what that predicts: "everything is too bright, darks and lights
    // don't work together." Settled roughly halfway between the two extremes rather than
    // either raw failure mode.
    float gateFloor = mix(0.20, 0.24, uDaylight);
    // Owner report: caves are way too bright, AND visibly change with time of day, when
    // nothing outside should reach them at all. Root cause: uSkyAmbient itself already
    // carries the FULL day/night swing (bright blue at noon, near-black at midnight), and
    // the old formula multiplied gateFloor straight against it, so a fully sky-occluded
    // surface (skyL=0, a real cave, not a shallow notch) tracked the sky's OWN cycle no
    // matter what constant gateFloor held, since it is a fraction of an already
    // time-varying quantity. No floor VALUE could have fixed this; the multiplication
    // itself was the bug. Damped the floor's own reference brightness specifically for
    // daylight, night's uSkyAmbient is already dim and needs no damping, so a deep,
    // fully-enclosed space now stays close to its NIGHT brightness through the day instead
    // of visibly lighting up at noon. The skyL=1 (full sky exposure) case is untouched: it
    // still resolves to the true, undamped uSkyAmbient, so open-air lighting does not move.
    vec3 floorAmbient = uSkyAmbient * mix(1.0, 0.3, uDaylight);
    // Owner report: underground looked nothing like the surface, with "no true darkness and
    // light". Every floor in here (the sky floor, the flat term, the backstop below) was there
    // for shade on the SURFACE, and all of them reached a sealed cave just the same: a cave wall
    // no lamp could see sat at the same grey as the walls one could, so a light had nothing to
    // fall off into. They fade out over the last few levels of sky now, which is shade nobody on
    // the surface ever stands in, and leave a sealed cave dark except where something lights it.
    float open = smoothstep(0.0, 0.35, skyL);
    // Sky light reaching into a hole used to fill it linearly with its level, so five blocks
    // inside a cave mouth still had two thirds of open-air ambient. Light that has turned
    // corners dies faster than that: same as before from about level 9 up, gone near the end.
    float skyFill = skyL * smoothstep(0.0, 0.6, skyL);
    // The flat term used to be an ungated constant, which reads as harmless next to a
    // daylight ambient of 0.26-0.55 but was 40-64% of the night ambient, a day-neutral
    // number that was, in practice, a night-only brightener. Faded out with daylight so it
    // still stops fully-enclosed faces (interior leaf pockets, a deep notch) crushing to
    // pure black in daylight, without lifting the whole night off the floor.
    vec3 ambient = mix(floorAmbient * gateFloor * open, uSkyAmbient, skyFill) +
                   vec3(0.035, 0.04, 0.05) * mix(0.25, 1.0, uDaylight) * open;
    // The emitter's own colour drives the tint now (M18), so the fixed warm multiplier
    // that used to supply it would double-tint. Keep a slight warm bias for the ember
    // feel and let the light itself carry the hue out to its falloff.
    vec3 torch = vec3(1.12, 1.0, 0.88) * (blockRGB * blockRGB * 2.1);

    // Per-corner brightness, interpolated across the face. It already carries the old AO
    // term (mesher bakes 0.68 + 0.32 * ao into it, so concave corners darken exactly as
    // much as they used to) multiplied by how much of the quad's light this corner
    // actually sees, that second half is what ramps a merged quad into its neighbour
    // instead of stepping. A quad with uniform corners is all 7s and shades flat, so open
    // terrain looks and costs the same as before.
    // An emitter's own face was only ever LIT by its own light, which made a lamp read as a
    // brightly lit block rather than a light, and left it hovering around the bloom
    // threshold, so the glow came and went with the face angle. Emission is added flat:
    // not scaled by kShade, AO or vShade, because none of those mean anything for a surface
    // that is producing the light rather than receiving it.
    //
    // The hue comes from the albedo, which for a lamp IS its swatch colour (the tiles are
    // generated from blocks::kLampRGB), so this needs no second copy of that table to drift
    // out of sync. Layer 4 is glow, 15..39 are the 25 lamp swatches, see texLayer() in
    // chunk.vert, which owns that mapping.
    // Leaf subsurface scattering. A leaf is thin and translucent: with the sun behind it,
    // light comes THROUGH rather than bouncing off, which is why a canopy lit from behind
    // glows instead of silhouetting flat black. The wrap term is the standard cheap stand-in
    // for a real transmission lobe, a back-facing lobe around -L, tightened with a power so
    // it only fires when you are actually looking toward the sun through the leaf.
    //
    // Layers 7 (leaves) and 40/41 (the cross-plane tuft and flower) are the thin geometry;
    // everything else is opaque and gets nothing. Scaled by skyAccess so leaves buried deep
    // inside a canopy, which no sunlight reaches, do not light up.
    vec3 transmit = vec3(0.0);
    if (uLeafSss > 0.0 && (isCutoutLayer(vLayer) || vDir >= 6u)) {
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
    // source. That distinction is the whole of #49, an emitter's own torch term is by far
    // the largest thing in its shading, so AO on it puts a dark band along the bottom of
    // every lamp sitting on the ground, exactly where the corners are most occluded. It
    // reads as the block being shadowed at its base while it is lighting everything around
    // it. kShade stays, so a lamp still reads as a cube rather than a flat sticker.
    float ownShade = mix(vShade, 1.0, emitMask);
    // Ambient is gated TWICE, and that is what crushes a shaded terrace face to black.
    // Once by skyL above (the voxel sky access), and again by vShade, which is AO times
    // that corner's share of the quad's light, and the light includes sky. So sky access
    // is applied on top of itself, squared, on exactly the surfaces that have least of it:
    // the inside faces of a notch, the risers of a terrace, anything under an overhang.
    //
    // AO genuinely does belong on ambient; the corner's light share does not, because
    // skyL already answered that question. They are baked into one 3-bit value at mesh
    // time and cannot be separated here, so the corner term is softened for the ambient
    // path rather than removed: at worst it takes 45% instead of everything. `direct`
    // keeps the full term, a face the sun does not reach should still go dark.
    float vShadeF = mix(vShade, 1.0, 0.7 * farK);
    float ambShade = mix(0.75, 1.0, vShadeF);
    // direct's OWN occlusion question, is the sun blocked here, is already answered by
    // `shadow` above, from the actual cascade, not a 4-voxel corner guess. Multiplying by
    // the full vShade on top of that double-asks the same question with a much coarser
    // answer: a terrace made of small steps has a real (correct) AO dip at every riser for
    // AMBIENT purposes, but none of those steps actually block the sun, which comes from
    // one direction high in the sky and does not care about a neighbour one block over. On
    // a long, gently terraced slope that showed up as a fine, regular hatch of dark/light
    // patches across ground the shadow map says is fully lit, confirmed by rendering with
    // shadows forced off entirely and finding the same pattern still there, unrelated to any
    // shadow-map behaviour. Softened the same shape as ambShade, just less aggressively,
    // since `shadow` is already doing the real work here and this is only damping residual
    // corner noise, not the thing that makes a genuine notch look recessed.
    float directShade = mix(0.7, 1.0, vShadeF);
    vec3 shading = direct * directShade + (ambient + cloudFill) * ambShade + torch * ownShade;
    // Hard backstop, not another adjustment to one contributing term: owner report, several
    // rounds running, is that ANY hard black reads as broken, regardless of which path
    // produced it (AO, the sky-access gate, a genuinely enclosed notch with no sun and
    // almost no sky). Rather than keep chasing individual cases, nothing lit at all is
    // allowed to render darker than this, full stop, dim and blue-tinted like the existing
    // flat ambient term above, so it reads as "in shadow" rather than as a flat grey floor.
    // Not underground, though: there the floor is a trace you can just make out rock by.
    shading = max(shading, mix(vec3(0.004, 0.0045, 0.006), vec3(0.05, 0.055, 0.065), open));
    outColor = vec4(albedo * mix(kShade[vDir], 0.88, farK) * shading + emission + transmit, 1.0);
}
