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

// screen-space-reflection water. reflects the opaque scene (trees, terrain) by ray
// marching the reflection ray against the depth buffer and sampling a snapshot of the
// opaque colour; falls back to a computed sky where the ray leaves the screen. fresnel
// drives both the reflection strength and the alpha, so looking down you see through to
// the seabed (the blend leaves the opaque pixel behind) and at grazing angles you see
// the mirror. animated wave normal, sharp sun specular.

in vec3 vWorldRel; // camera-relative world position
flat in uint vDir;
in float vSky; // per-corner now, so a shaded lake edge ramps instead of stepping
in float vStrength; // 0..1 how much swell this part of the surface has (mesher: shoaling, shores)

layout(location = 0) out vec4 outColor;

layout(location = 0) uniform mat4 uViewProj; // occupies 0..3 (same as the vertex stage)
layout(location = 4) uniform float uTime;
layout(location = 5) uniform vec3 uSunL;
layout(location = 6) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 7) uniform float uNear;
layout(location = 8) uniform vec3 uSunColor;   // day/night sun radiance
layout(location = 9) uniform vec3 uSkyAmbient; // day/night sky fill, same one the terrain uses
layout(location = 10) uniform float uShadowsOn;
layout(location = 11) uniform mat4 uCascadeVp[3];  // consumes 11..22
layout(location = 23) uniform vec3 uCascadeSplit;  // radial split distances
layout(location = 24) uniform vec3 uCascadeTexel;  // world units per shadow texel
layout(location = 40) uniform vec3 uWaterLevel; // keeps its brightness under the land's curve
layout(location = 25) uniform float uDaylight;     // 1 full day, 0 after dusk
layout(location = 43) uniform vec3 uHandLight;     // a lamp in the player's hand, levels 0..15 (chunk.frag)

layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;

layout(binding = 2) uniform sampler2D uSceneColor; // opaque snapshot
layout(binding = 3) uniform sampler2D uSceneDepth;
// The same depth before the animals went in. Only the shoreline foam reads it: a fish just under
// the surface is thin water to uSceneDepth, and foam keyed off that ringed every fish in white.
layout(binding = 4) uniform sampler2D uTerrainDepth;

const vec3 kNormal[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));

float hash21(vec2 p);
float vnoiseW(vec2 p, float n);

#include "waves.glsl"
#include "sky.glsl"
#define CLOUD_BIAS_LOCATION 44
#include "clouds.glsl"
#define CLOUD_WEATHER_BINDING 6
#include "cloudfield.glsl"
layout(location = 45) uniform float uOvercast;  // rain greys the sky it reflects, as composite.frag's
layout(location = 46) uniform float uCloudsOn;
layout(location = 47) uniform vec3 uCloudCam;   // camera x, z modulo the weather map's span, and y
// The camera's xz modulo 512 (render/WaterWaves.hpp kWrap). The chop and the foam are value noise
// whose lattice repeats every 512 blocks, so camera-relative position plus this is world position
// to that noise: the pattern stays put as the camera moves (it used to travel along with it, the
// whole surface sliding under a walking player) without a large coordinate ever reaching a float.
layout(location = 34) uniform vec2 uCamWrap;

// Slope of the water surface, d(height)/d(xz), at a camera-relative point. Two parts that do not
// share a period, which is the whole trick: pure sines summed make a fixed interference pattern,
// and on an open lake that reads as diagonal corduroy rather than as water. The swell carries the
// shape, a scrolling noise field the chop, and because the noise is not periodic against the
// swell nothing tiles.
//
// The swell's slope here is the slope of the surface water.tese actually moves (same waves, same
// strength), so near the camera the shading and the geometry agree. Past the fade the geometry is
// flat again and only this keeps the swell, so far water does not turn to glass.
//
// Each wave is faded by its own wavelength against the pixel's footprint (fwidth, measured once
// in main()). Owner report, live screenshots: wide lake views showed dark contour-line streaks
// and wedges sweeping the whole surface, because perspective packs many periods of a wave into a
// few pixels and the slope then swings at random; the chop always had a distance fade for that
// reason, the swell had none. A wave goes once it is under a few pixels a period.
vec2 waterSlope(vec2 rel, float strength, float detail, float footprint) {
    float scale = (uWaveScale > 0.0 ? uWaveScale : 0.30) * strength;
    vec2 g = vec2(0.0);
    for (int i = 0; i < 4; ++i) {
        float len = 6.2831853 / uWave[i].z;
        float aa = 1.0 - smoothstep(0.2 * len, 0.6 * len, footprint);
        float th = uWave[i].z * dot(uWave[i].xy, rel) + uWavePhase[i];
        float e = exp(uWaveShape.x * (sin(th) - 1.0));
        g += (uWave[i].w * uWaveShape.w * e * uWaveShape.x * cos(th) * uWave[i].z * aa) * uWave[i].xy;
    }
    g *= scale;

    // Chop: central differences over a scrolling noise field, two octaves drifting in
    // different directions so the pattern never settles into a repeat. Scaled up six times
    // into slope: it is detail the eye reads off the highlights, not height anyone sees.
    //
    // `detail` fades the chop out with distance, and it is not optional. A ripple smaller
    // than a pixel makes the 340-exponent specular fire on essentially random normals, so
    // far water turns into a field of crawling white sparkles. Fading it is also the honest
    // answer physically, you cannot resolve centimetre chop 200 blocks away.
    if (detail <= 0.004)
        return g;
    vec2 p = rel + uCamWrap;
    const float e = 0.35;
    // frequencies are n / 512 so each octave's lattice repeats exactly every 512 blocks
    vec2 q1 = p * 0.5625 + vec2(uTime * 0.16, uTime * 0.11);
    vec2 q2 = p * 1.5625 + vec2(-uTime * 0.27, uTime * 0.19);
    vec2 dh = vec2(0.0);
    for (int a = 0; a < 2; ++a) {
        vec2 q = a == 0 ? q1 : q2;
        float n = a == 0 ? 288.0 : 800.0;
        float amp = a == 0 ? 0.055 : 0.022;
        float sc = a == 0 ? 0.5625 : 1.5625;
        float c = vnoiseW(q, n);
        float gx = vnoiseW(q + vec2(e, 0.0), n) - c;
        float gy = vnoiseW(q + vec2(0.0, e), n) - c;
        dh += vec2(gx, gy) * (amp * sc / e) * detail;
    }
    return g + dh * 6.0;
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
// Value noise whose lattice wraps every n cells.
float vnoiseW(vec2 p, float n) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(mod(i, n)), b = hash21(mod(i + vec2(1, 0), n));
    float c = hash21(mod(i + vec2(0, 1), n)), d = hash21(mod(i + vec2(1, 1), n));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// p in foam lattice units (1.3125 a block): the two octaves repeat every 672 and 1344 cells,
// both exactly 512 blocks
float fbm2(vec2 p) { return 0.6 * vnoiseW(p, 672.0) + 0.4 * vnoiseW(p * 2.0 + 3.7, 1344.0); }

// The sky seen along d from the water at P (camera-relative).
vec3 skyColor(vec3 d, vec3 P) {
    // uDaylight, not uSunL.y: uSunL points at the MOON after dusk, so deriving the day
    // factor from it would make a midnight lake reflect a bright blue noon sky.
    vec3 col = skyGradient(d, uDaylight > 0.0 ? uSunL : -uSunL, uDaylight, uOvercast);
    float sd = max(dot(d, uSunL), 0.0);
    // "is the thing lighting us above the horizon", true for the moon too, which is what
    // puts a moon glint on the water instead of nothing.
    float above = smoothstep(-0.03, 0.06, uSunL.y);
    float cover = 0.0;
    if (uCloudsOn > 0.5 && d.y > 0.01) {
        // The cloud layer where the reflected ray meets it, from the same weather map and cover the
        // sky's clouds are marched from, lit like their undersides: a lake mirrors the clouds over it.
        vec2 hit = uCloudCam.xz + P.xz + d.xz / d.y * (kCloudMid - uCloudCam.y - P.y);
        cover = cloudCoverage(cloudWeather(hit, uTime).r) * smoothstep(0.02, 0.12, d.y);
        vec3 lit = uSkyAmbient * 1.5 + uSunColor * (0.3 + 0.5 * pow(sd, 4.0)) * smoothstep(-0.02, 0.15, uSunL.y);
        col = mix(col, lit * mix(0.3, 1.0, smoothstep(0.0, 0.25, uDaylight)), cover * 0.85);
    }
    col += uSunColor * smoothstep(0.9985, 0.9995, sd) * 3.0 * above * (1.0 - cover) * (1.0 - uOvercast); // disc
    return col;
}

// Cascade shadow, the same three the terrain samples. Water had NO shadow lookup at all:
// every sun term was gated on vSky, which is 4-bit per-voxel sky ACCESS. That is a
// geometric value on a 16-level grid, so the specular and the surface brightness stepped
// from block to block, the blocky sun lighting. A shadow lookup is smooth, moves with the
// sun, and is what the terrain has always used.
float waterShadow(vec3 pos, vec3 N) {
    if (uShadowsOn < 0.5)
        return 1.0;
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    // Water's normal is near-vertical (up, plus a little wave perturbation), so a low,
    // near-horizontal sun grazes it exactly the way it grazes a steep terrain riser, this
    // had zero compensation for that until now, unlike chunk.frag's sampleCascade() (#68),
    // which is the other half of why the owner's "hard shadows" report showed up on water too.
    float ndl = max(dot(N, uSunL), 0.0);
    float slope = 1.0 / max(ndl, 0.2);
    vec4 sc = uCascadeVp[c] * vec4(pos + N * uCascadeTexel[c] * 1.5 * min(slope, 3.0), 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float z = sc.z - 0.0016 * min(slope, 2.5);
    // 4 taps, rotated per pixel for the same reason chunk.frag's sampleCascade() now rotates
    // its 16: fixed corner taps sample identical offsets everywhere, which aliases against
    // the shadow map's own voxel-scale depth edges into a hard, coherent patch instead of
    // noise. Rotating breaks that spatial correlation.
    float rotA = fract(sin(dot(gl_FragCoord.xy, vec2(35.17, 91.73))) * 27321.58) * 6.2831853;
    mat2 rot = mat2(cos(rotA), sin(rotA), -sin(rotA), cos(rotA));
    float t = 1.5 / VX_SHADOW_RES;
    float sh = 0.25 * (texture(uShadowMap, vec4(uv + rot * vec2(t, t), float(c), z)) +
                       texture(uShadowMap, vec4(uv + rot * vec2(-t, t), float(c), z)) +
                       texture(uShadowMap, vec4(uv + rot * vec2(t, -t), float(c), z)) +
                       texture(uShadowMap, vec4(uv + rot * vec2(-t, -t), float(c), z)));
    return mix(sh, 1.0, smoothstep(uCascadeSplit.z * 0.9, uCascadeSplit.z, d));
}

// How much sky this water can see. A reflection that finds nothing falls back to the sky, which in a
// sealed cave turned every pool bright blue.
float gSkyVis = 1.0;

// march R from world-rel point P, return reflected colour (scene hit or sky)
vec3 reflection(vec3 P, vec3 R, float jitter) {
    // Start the march a fraction of a step along, varying per pixel. Every ray used to
    // begin at the same offset and grow by the same ratio, so against a flat water plane
    // the sample depths landed in lockstep across the screen and the acceptance window
    // resolved in bands, horizontal stripes that slid as the view angle changed the ray
    // slope. Breaking the phase per pixel turns that banding into noise the eye reads as
    // surface detail instead of a grid.
    float stepLen = 0.6;
    vec3 sp = P + R * (stepLen * jitter);
#if   VX_QUALITY == 0
    const int kMarch = 8;
#elif VX_QUALITY == 1
    const int kMarch = 14;
#elif VX_QUALITY == 3
    const int kMarch = 40;
#else
    const int kMarch = 28;
#endif
    // Owner report: distant reflections came out as horizontal lines, island and sky in
    // alternating rows. A sample used to count as a hit only within 2.5 blocks behind the
    // surface, but the steps grow to tens of blocks out at a far island, so most rays jumped
    // from in front of it to well behind it in one step, missed, and fell back to sky; which
    // rows landed inside the window followed the swell crests. Now a hit is the ray crossing:
    // in front of the scene at the last sample, behind it at this one, by no more than this
    // step can explain (still bounded, so a ray passing behind a thin trunk does not catch it).
    bool wasFront = true; // starts at the water, in front of whatever is behind it
    for (int i = 0; i < kMarch; ++i) {
        sp += R * stepLen;
        vec4 clip = uViewProj * vec4(sp, 1.0);
        if (clip.w <= 0.0)
            break;
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
            break;
        float rawD = texture(uSceneDepth, uv).r;
        if (rawD > 0.0) {
            float sceneZ = -uNear / rawD;   // view-space z of the scene at uv (negative)
            float marchZ = -clip.w;         // view-space z of the marched point
            float behind = sceneZ - marchZ; // > 0: the marched point is behind the scene
            bool crossed = behind > 0.0 && wasFront && behind < max(2.5, stepLen * 1.5);
            wasFront = behind <= 0.0;
            if (crossed) {
                // Binary-refine the crossing. A linear march only knows the hit to within
                // one (growing) step, and at grazing angles that step is metres long, the
                // reflection then samples well past the silhouette it should have stopped
                // at, which is what smeared reflected trees into vertical streaks.
                vec3 lo = sp - R * stepLen, hi = sp;
                for (int k = 0; k < 5; ++k) {
                    vec3 mid = (lo + hi) * 0.5;
                    vec4 mc = uViewProj * vec4(mid, 1.0);
                    vec2 muv = mc.xy / mc.w * 0.5 + 0.5;
                    float mz = -uNear / max(texture(uSceneDepth, muv).r, 1e-7);
                    if (-mc.w < mz) hi = mid; else lo = mid;
                }
                vec4 hc = uViewProj * vec4(hi, 1.0);
                uv = hc.xy / hc.w * 0.5 + 0.5;
                // fade near screen edges to hide the SSR cutoff
                vec2 e = smoothstep(0.0, 0.12, uv) * smoothstep(0.0, 0.12, 1.0 - uv);
                float edge = e.x * e.y;
                return mix(skyColor(R, P) * gSkyVis, texture(uSceneColor, uv).rgb, edge);
            }
        } else {
            wasFront = true; // sky here: nothing to be behind
        }
        stepLen *= 1.18; // grow steps: dense near the surface, coarse far away
    }
    return skyColor(R, P) * gSkyVis;
}

void main() {
    vec3 N = kNormal[vDir];
    vec2 g = vec2(0.0);
    // Separate, much gentler normal for the REFLECTION ray only (see reflection() call
    // below). Shading wants the full-strength ripple normal, that's what makes the glint
    // and fresnel read as choppy water. But an SSR ray-marcher is only as stable as the
    // direction it's given: two neighbouring pixels with noticeably different wave-perturbed
    // normals can point at completely different march targets, one landing a hit and the
    // other missing and falling back to sky, and since the miss/hit boundary is then driven
    // by the same noise field as the waves, it comes out as the same cellular/blobby shape,
    // not uniform noise. Reported live as "black spots at some angles" (measured: not
    // literal black, dark blue-gray sky-fallback tone, ~RGB 30-45). Damping the normal fed
    // to the reflection ray specifically, not the shading normal, keeps neighbouring rays
    // coherent enough to agree on hit/miss while leaving the actual water surface look alone.
    vec3 Nrefl = kNormal[vDir];
    vec2 breakNow = vec2(0.0);
    if (vDir == 2u) { // organic ripple on the top surface
        // Chop detail falls off with distance; see waterSlope. Squared so it holds up close
        // and then drops away quickly rather than lingering as a shimmer band.
        float fade = exp(-length(vWorldRel) * 0.016);
        g = waterSlope(vWorldRel.xz, vStrength, fade * fade, length(fwidth(vWorldRel.xz)));
        // A breaker tilts the water toward the shore, the way its distance map says.
        float seaD = seaDistance(vWorldRel.xz);
        breakNow = breaker(vWorldRel.xz, seaD);
        if (uWaveScale > 0.0 && seaD < 80.0 && seaD > -6.0) {
            const float kE = 3.0;
            vec2 cuv = (vWorldRel.xz - uCoastXf.xy) * uCoastXf.z;
            vec2 ts = vec2(1.0) / vec2(textureSize(uCoast, 0));
            vec2 dirD = -vec2(textureLod(uCoast, cuv + vec2(ts.x, 0.0), 0.0).r - textureLod(uCoast, cuv - vec2(ts.x, 0.0), 0.0).r,
                              textureLod(uCoast, cuv + vec2(0.0, ts.y), 0.0).r - textureLod(uCoast, cuv - vec2(0.0, ts.y), 0.0).r);
            dirD /= max(length(dirD), 1e-3);
            float slope = (breaker(vWorldRel.xz, seaD + kE).x - breaker(vWorldRel.xz, seaD - kE).x) / (2.0 * kE);
            g += clamp(dirD * slope * uWaveScale * vStrength * 0.8, vec2(-0.3), vec2(0.3));
        }
        N = normalize(vec3(-g.x, 1.0, -g.y));
        Nrefl = normalize(vec3(-g.x * 0.25, 1.0, -g.y * 0.25));
    }
    vec3 V = normalize(-vWorldRel); // camera at the origin
    // Schlick off water's real F0 (0.02) is physically right and looks like jelly: from
    // above you get 2% reflection and 98% murky refraction, so the surface reads as a
    // translucent solid rather than a mirror. Real water sells itself on reflection at
    // every angle, and the look people mean by "raytraced water" is a much higher floor.
    // This is a deliberate art choice, not a physical one, the grazing end still goes to
    // 1.0, so the Fresnel falloff is intact, the floor is just lifted.
    const float kBaseReflect = 0.34;
    // Owner report, screenshots from directly overhead: fine dark contour-line tracery
    // covering the WHOLE water surface at every distance and angle, not worse far away,
    // ruled out the swell/chop antialiasing fix above (which only matters at distance) and
    // pointed somewhere that depends on the per-pixel wave normal directly, at any range.
    // Found it: #72 damped the REFLECTION RAY direction (Nrefl, 25% strength) to stop SSR
    // hit/miss noise, but this fresnel weight, which decides how much of that reflection
    // shows through at all, was left reading the FULL-strength, undamped wave normal N.
    // Reflection content is often darker than refraction (a distant tree line, a dim patch
    // of sky, a miss falling back to whatever's there), so with fresnel flickering between
    // "mostly clear refraction" and "mostly whatever SSR happened to hit" on every fine
    // ripple, the wave's own contour gets traced out in alternating light/dark, present at
    // any distance because it depends on the local normal, not on how well the noise field
    // is resolved. Same fix as #72, applied to the piece #72 left untouched: use the damped
    // Nrefl here too, so the BLEND WEIGHT is stable even though Nrefl already makes the ray
    // DIRECTION stable. N itself is untouched for ndl/specular, which is what actually gives
    // water its choppy sparkle, this only stops the reflection blend from chasing every
    // ripple.
    float fres = mix(kBaseReflect, 1.0, pow(1.0 - max(dot(Nrefl, V), 0.0), 5.0));

    vec2 res = vec2(textureSize(uSceneDepth, 0));
    vec2 suv = gl_FragCoord.xy / res;

    // water thickness at this pixel (surface vs seabed, view space) -> absorption + foam
    float bedZ = -uNear / max(texture(uSceneDepth, suv).r, 1e-7);
    float surfZ = -(uViewProj * vec4(vWorldRel, 1.0)).w;
    float thickness = max(surfZ - bedZ, 0.0);

    // refraction: sample the seabed nudged by the wave slope. shrink the offset in shallow
    // water so the shoreline doesn't smear, and undo it if it grabbed a nearer (foreground)
    // pixel that isn't actually under the water.
    vec2 ruv = clamp(suv + g * 0.01 * clamp(thickness * 0.5, 0.0, 1.0), vec2(0.001), vec2(0.999));
    if (-uNear / max(texture(uSceneDepth, ruv).r, 1e-7) > surfZ)
        ruv = suv;
    vec3 refr = texture(uSceneColor, ruv).rgb;

    // How lit this patch of water is. sun = the cascade (smooth, sun-following); ambient =
    // the same sky fill the terrain gets. vSky survives only as a soft occlusion factor for
    // water under an overhang, never as the thing that shades the sun, which is what made
    // the lighting step from block to block.
    float shadow = waterShadow(vWorldRel, N);
    if (uCloudsOn > 0.5) {
        // a cloud between it and the sun takes the glint and the sunlit body with it (lighting.glsl's cloudCover)
        vec2 at = uCloudCam.xz + vWorldRel.xz + uSunL.xz * ((kCloudMid - 64.0) / max(uSunL.y, 0.05));
        shadow *= 1.0 - 0.9 * smoothstep(0.2, 0.7, cloudCoverage(cloudWeather(at, uTime).r));
    }
    float ndl = max(dot(N, uSunL), 0.0);
    float occl = 0.35 + 0.65 * vSky;            // gentle: 4-bit voxel light, so keep it soft
    gSkyVis = smoothstep(0.02, 0.45, vSky);
    vec3 sunLit = uSunColor * (ndl * shadow * gSkyVis);
    vec3 ambLit = uSkyAmbient * occl * mix(0.04, 1.0, gSkyVis);
    // Daylight scalar for anything that must go dark at night. Supplied by the CPU now:
    // uSunL points at the moon after dusk, so it can no longer answer "how light is it".
    float daylight = uDaylight;

    // Beer-Lambert absorption: shallow shows the bed, deep goes blue-green.
    // Over the far terrain's own sea (alpha 0) this is deep water, whatever the depth buffer says.
    float overFar = 1.0 - texture(uSceneColor, suv).a;
    // Each colour lost at its own rate, the same as looking through it from below (composite.frag).
    vec3 keep = exp(-vec3(0.20, 0.09, 0.05) * thickness) * (1.0 - overFar);
    float absorb = 1.0 - dot(keep, vec3(0.3333));
    vec3 tint = mix(vec3(0.10, 0.26, 0.40), vec3(0.015, 0.06, 0.18), absorb);
    // The tint was a CONSTANT colour scaled only by vSky, and vSky is geometric, it does
    // not drop when the sun sets. So deep water kept full daylight brightness after dark and
    // the lake read as luminous. It is a body colour: light it like one. (The foam had the
    // same bug and was fixed years earlier; the water itself never was.)
    // The trailing constant is the same class of bug the tint itself had: ungated, it is
    // negligible against a daylight ambient and roughly half the night one, so it quietly
    // held the lake off the floor after dark. Faded with daylight for the same reason.
    vec3 waterLight = ambLit + sunLit * 0.35 + vec3(0.015, 0.02, 0.028) * mix(0.25, 1.0, daylight);
    // A lamp in the hand lights the water like the blocks around it, and shows in it as a soft glint.
    float handLevel = max(uHandLight.r, max(uHandLight.g, uHandLight.b));
    float hand = max(handLevel - length(vWorldRel), 0.0) / 15.0;
    vec3 handLit = vec3(1.12, 1.0, 0.88) * (uHandLight / max(handLevel, 1.0)) * hand * hand * 1.2;
    waterLight += handLit;
    refr = refr * occl * keep + tint * waterLight * uWaterLevel * (1.0 - keep);

    float jitter = hash21(gl_FragCoord.xy + fract(uTime) * 17.0);
    // The reflection carries its own light (it is a picture of the lit world / the sky), so
    // it only wants the occlusion term, not a second helping of daylight.
    vec3 refl = reflection(vWorldRel, reflect(-V, Nrefl), jitter) * occl * uWaterLevel;

    // two-lobe glint: a tight hot core plus a broad sheen, which is what separates a wet
    // mirror from a plastic-looking one under a single narrow highlight
    vec3 H = normalize(uSunL + V);
    float ndh = max(dot(N, H), 0.0);
    // Shadowed by the cascade rather than gated on voxel sky light: a glint that survives
    // into a tree's shadow is the giveaway that the sun term was never really shadowed.
    float spec = (pow(ndh, 340.0) * 2.6 + pow(ndh, 34.0) * 0.28) * shadow * occl * gSkyVis;

    vec3 col = mix(refr, refl, fres) + uSunColor * spec; // glint fades at night
    // Over the far terrain's own sea, at the edge of the streamed world, be that sea: it has its own
    // colour and its own sky in it already, and a second helping of both here was a pale ring.
    col = mix(col, texture(uSceneColor, suv).rgb, overFar);
    // the lamp's reflection: the light is at the eye, so it sits where the surface faces straight back
    col += handLit * (pow(max(dot(N, V), 0.0), 60.0) * 0.6);

    // Shoreline foam: thin water gets an animated fringe. It has to be LIT like everything
    // else, as a constant white it stayed full brightness after dark, so every shoreline
    // glowed against black water, which is the "water is luminescent at night" and the
    // white edge wherever water meets a block. Tie it to the sky light reaching this
    // surface and to how much sun there is, so at night it goes to the same near-black the
    // rest of the water does.
    float bedZ0 = -uNear / max(texture(uTerrainDepth, suv).r, 1e-7);
    float shore = 1.0 - smoothstep(0.0, 1.6, max(surfZ - bedZ0, 0.0));
    float fn = fbm2((vWorldRel.xz + uCamWrap) * 1.3125 + vec2(uTime * 0.3, -uTime * 0.2));
    float foam = smoothstep(0.55, 0.95, shore * (0.55 + 0.6 * fn)) * (1.0 - overFar);
    // The white of a wave going over, broken into lace by the same noise.
    float lace = fbm2((vWorldRel.xz + uCamWrap) * 0.66 + vec2(uTime * 0.11, uTime * 0.07));
    foam = max(foam, smoothstep(0.26, 0.66, breakNow.y * (0.45 + 0.9 * lace)) * min(uWaveScale / 0.2, 1.0));
    vec3 foamCol = vec3(0.92, 0.96, 1.0) * occl * (0.10 + 0.90 * daylight) * max(gSkyVis, 0.03);
    col = mix(col, foamCol, foam);

    outColor = vec4(col, 1.0); // fully composited here (refraction sampled directly)
}
