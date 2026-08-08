#version 460 core

// screen-space-reflection water. reflects the opaque scene (trees, terrain) by ray
// marching the reflection ray against the depth buffer and sampling a snapshot of the
// opaque colour; falls back to a computed sky where the ray leaves the screen. fresnel
// drives both the reflection strength and the alpha, so looking down you see through to
// the seabed (the blend leaves the opaque pixel behind) and at grazing angles you see
// the mirror. animated wave normal, sharp sun specular.

in vec3 vWorldRel; // camera-relative world position
flat in uint vDir;
in float vSky; // per-corner now, so a shaded lake edge ramps instead of stepping

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

layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;

layout(binding = 2) uniform sampler2D uSceneColor; // opaque snapshot
layout(binding = 3) uniform sampler2D uSceneDepth;

const vec3 kNormal[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));

float hash21(vec2 p);
float vnoise(vec2 p);

// Wave slope. Two scales that do not share a period, which is the whole trick: four pure
// sines summed produce a fixed interference pattern, and on an open lake that reads as
// diagonal corduroy rolling across the surface rather than as water. A big directional
// swell carries the shape, a scrolling noise field carries the chop, and because the noise
// is not periodic against the sines nothing tiles.
//
// Returns d(height)/d(xz). The normal is built from it in main().
vec2 waveGrad(vec2 p, float detail) {
    vec2 g = vec2(0.0);

    // Swell: three long directional waves. Kept few and low frequency — this is the part
    // the eye reads as "which way the water is moving".
    const vec4 W[3] = vec4[3](vec4(0.80, 0.20, 0.10, 0.055), vec4(-0.30, 0.95, 0.16, 0.038),
                              vec4(0.55, -0.72, 0.23, 0.024));
    const float sp[3] = float[3](0.55, 0.78, 1.05);
    for (int i = 0; i < 3; ++i) {
        vec2 d = normalize(W[i].xy);
        float ph = dot(d, p) * W[i].z + uTime * sp[i];
        g += d * W[i].z * W[i].w * cos(ph);
    }

    // Chop: central differences over a scrolling noise field, two octaves drifting in
    // different directions so the pattern never settles into a repeat.
    //
    // `detail` fades the chop out with distance, and it is not optional. A ripple smaller
    // than a pixel makes the 340-exponent specular fire on essentially random normals, so
    // far water turns into a field of crawling white sparkles. Fading it is also the honest
    // answer physically — you cannot resolve centimetre chop 200 blocks away.
    if (detail <= 0.004)
        return g;
    const float e = 0.35;
    vec2 q1 = p * 0.55 + vec2(uTime * 0.16, uTime * 0.11);
    vec2 q2 = p * 1.60 + vec2(-uTime * 0.27, uTime * 0.19);
    float h = 0.0;
    vec2 dh = vec2(0.0);
    for (int a = 0; a < 2; ++a) {
        vec2 q = a == 0 ? q1 : q2;
        float amp = a == 0 ? 0.055 : 0.022;
        float sc = a == 0 ? 0.55 : 1.60;
        float c = vnoise(q);
        float gx = vnoise(q + vec2(e, 0.0)) - c;
        float gy = vnoise(q + vec2(0.0, e)) - c;
        dh += vec2(gx, gy) * (amp * sc / e) * detail;
        h += c;
    }
    return g + dh;
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1, 0));
    float c = hash21(i + vec2(0, 1)), d = hash21(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm2(vec2 p) { return 0.6 * vnoise(p) + 0.4 * vnoise(p * 2.1 + 3.7); }

vec3 skyColor(vec3 d) {
    float day = clamp(uSunL.y * 3.0 + 0.12, 0.0, 1.0);
    float up = clamp(d.y, 0.0, 1.0);
    vec3 col = mix(vec3(0.66, 0.78, 0.92), vec3(0.14, 0.34, 0.80), pow(up, 0.5)) * day +
               vec3(0.02, 0.03, 0.06) * (1.0 - day); // dim to a night sky
    float sd = max(dot(d, uSunL), 0.0);
    float above = smoothstep(-0.03, 0.06, uSunL.y);
    col += vec3(1.0, 0.55, 0.28) * pow(sd, 8.0) * 0.6 * above;
    col += uSunColor * smoothstep(0.9985, 0.9995, sd) * 3.0 * above;
    return col;
}

// Cascade shadow, the same three the terrain samples. Water had NO shadow lookup at all:
// every sun term was gated on vSky, which is 4-bit per-voxel sky ACCESS. That is a
// geometric value on a 16-level grid, so the specular and the surface brightness stepped
// from block to block — the blocky sun lighting. A shadow lookup is smooth, moves with the
// sun, and is what the terrain has always used.
float waterShadow(vec3 pos, vec3 N) {
    if (uShadowsOn < 0.5)
        return 1.0;
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    vec4 sc = uCascadeVp[c] * vec4(pos + N * uCascadeTexel[c] * 1.5, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float z = sc.z - 0.0016;
    // 4 taps: water is a broad flat surface, so a single tap aliases along the cascade grid
    float t = 1.5 / 1536.0;
    float sh = 0.25 * (texture(uShadowMap, vec4(uv + vec2(t, t), float(c), z)) +
                       texture(uShadowMap, vec4(uv + vec2(-t, t), float(c), z)) +
                       texture(uShadowMap, vec4(uv + vec2(t, -t), float(c), z)) +
                       texture(uShadowMap, vec4(uv + vec2(-t, -t), float(c), z)));
    return mix(sh, 1.0, smoothstep(uCascadeSplit.z * 0.9, uCascadeSplit.z, d));
}

// march R from world-rel point P, return reflected colour (scene hit or sky)
vec3 reflection(vec3 P, vec3 R, float jitter) {
    // Start the march a fraction of a step along, varying per pixel. Every ray used to
    // begin at the same offset and grow by the same ratio, so against a flat water plane
    // the sample depths landed in lockstep across the screen and the acceptance window
    // resolved in bands — horizontal stripes that slid as the view angle changed the ray
    // slope. Breaking the phase per pixel turns that banding into noise the eye reads as
    // surface detail instead of a grid.
    float stepLen = 0.6;
    vec3 sp = P + R * (stepLen * jitter);
    for (int i = 0; i < 28; ++i) {
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
            if (marchZ < sceneZ && marchZ > sceneZ - 2.5) {
                // Binary-refine the crossing. A linear march only knows the hit to within
                // one (growing) step, and at grazing angles that step is metres long — the
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
                return mix(skyColor(R), texture(uSceneColor, uv).rgb, edge);
            }
        }
        stepLen *= 1.18; // grow steps: dense near the surface, coarse far away
    }
    return skyColor(R);
}

void main() {
    vec3 N = kNormal[vDir];
    vec2 g = vec2(0.0);
    if (vDir == 2u) { // organic ripple on the top surface
        // Detail falls off with distance; see waveGrad. Squared so it holds up close and
        // then drops away quickly rather than lingering as a shimmer band.
        float fade = exp(-length(vWorldRel) * 0.016);
        g = waveGrad(vWorldRel.xz, fade * fade);
        N = normalize(vec3(-g.x * 6.0, 1.0, -g.y * 6.0));
    }
    vec3 V = normalize(-vWorldRel); // camera at the origin
    // Schlick off water's real F0 (0.02) is physically right and looks like jelly: from
    // above you get 2% reflection and 98% murky refraction, so the surface reads as a
    // translucent solid rather than a mirror. Real water sells itself on reflection at
    // every angle, and the look people mean by "raytraced water" is a much higher floor.
    // This is a deliberate art choice, not a physical one - the grazing end still goes to
    // 1.0, so the Fresnel falloff is intact, the floor is just lifted.
    const float kBaseReflect = 0.34;
    float fres = mix(kBaseReflect, 1.0, pow(1.0 - max(dot(N, V), 0.0), 5.0));

    vec2 res = vec2(textureSize(uSceneDepth, 0));
    vec2 suv = gl_FragCoord.xy / res;

    // water thickness at this pixel (surface vs seabed, view space) -> absorption + foam
    float bedZ = -uNear / max(texture(uSceneDepth, suv).r, 1e-7);
    float surfZ = -(uViewProj * vec4(vWorldRel, 1.0)).w;
    float thickness = max(surfZ - bedZ, 0.0);

    // refraction: sample the seabed nudged by the wave slope. shrink the offset in shallow
    // water so the shoreline doesn't smear, and undo it if it grabbed a nearer (foreground)
    // pixel that isn't actually under the water.
    vec2 ruv = clamp(suv + g * 0.06 * clamp(thickness * 0.5, 0.0, 1.0), vec2(0.001), vec2(0.999));
    if (-uNear / max(texture(uSceneDepth, ruv).r, 1e-7) > surfZ)
        ruv = suv;
    vec3 refr = texture(uSceneColor, ruv).rgb;

    // How lit this patch of water is. sun = the cascade (smooth, sun-following); ambient =
    // the same sky fill the terrain gets. vSky survives only as a soft occlusion factor for
    // water under an overhang — never as the thing that shades the sun, which is what made
    // the lighting step from block to block.
    float shadow = waterShadow(vWorldRel, N);
    float ndl = max(dot(N, uSunL), 0.0);
    float occl = 0.35 + 0.65 * vSky;            // gentle: 4-bit voxel light, so keep it soft
    vec3 sunLit = uSunColor * (ndl * shadow);
    vec3 ambLit = uSkyAmbient * occl;
    // Daylight scalar for anything that must go dark at night.
    float daylight = clamp(uSunL.y * 3.0 + 0.12, 0.0, 1.0);

    // Beer-Lambert absorption: shallow shows the bed, deep goes blue-green.
    float absorb = 1.0 - exp(-thickness * 0.18);
    vec3 tint = mix(vec3(0.12, 0.30, 0.36), vec3(0.02, 0.09, 0.16), absorb);
    // The tint was a CONSTANT colour scaled only by vSky, and vSky is geometric — it does
    // not drop when the sun sets. So deep water kept full daylight brightness after dark and
    // the lake read as luminous. It is a body colour: light it like one. (The foam had the
    // same bug and was fixed years earlier; the water itself never was.)
    vec3 waterLight = ambLit + sunLit * 0.35 + vec3(0.015, 0.02, 0.028);
    refr = mix(refr * occl, tint * waterLight, absorb);

    float jitter = hash21(gl_FragCoord.xy + fract(uTime) * 17.0);
    // The reflection carries its own light (it is a picture of the lit world / the sky), so
    // it only wants the occlusion term, not a second helping of daylight.
    vec3 refl = reflection(vWorldRel, reflect(-V, N), jitter) * occl;

    // two-lobe glint: a tight hot core plus a broad sheen, which is what separates a wet
    // mirror from a plastic-looking one under a single narrow highlight
    vec3 H = normalize(uSunL + V);
    float ndh = max(dot(N, H), 0.0);
    // Shadowed by the cascade rather than gated on voxel sky light: a glint that survives
    // into a tree's shadow is the giveaway that the sun term was never really shadowed.
    float spec = (pow(ndh, 340.0) * 2.6 + pow(ndh, 34.0) * 0.28) * shadow * occl;

    vec3 col = mix(refr, refl, fres) + uSunColor * spec; // glint fades at night

    // Shoreline foam: thin water gets an animated fringe. It has to be LIT like everything
    // else - as a constant white it stayed full brightness after dark, so every shoreline
    // glowed against black water, which is the "water is luminescent at night" and the
    // white edge wherever water meets a block. Tie it to the sky light reaching this
    // surface and to how much sun there is, so at night it goes to the same near-black the
    // rest of the water does.
    float shore = 1.0 - smoothstep(0.0, 1.6, thickness);
    float fn = fbm2(vWorldRel.xz * 1.3 + vec2(uTime * 0.3, -uTime * 0.2));
    float foam = smoothstep(0.55, 0.95, shore * (0.55 + 0.6 * fn));
    vec3 foamCol = vec3(0.92, 0.96, 1.0) * occl * (0.10 + 0.90 * daylight);
    col = mix(col, foamCol, foam);

    outColor = vec4(col, 1.0); // fully composited here (refraction sampled directly)
}
