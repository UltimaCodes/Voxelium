// Shared cloud field. Included by composite.frag (which raymarches it for the sky) and by
// chunk.frag (which samples it once per fragment for cloud shadows). One copy, because a
// second one would drift and the symptom would be shadows that do not match the clouds
// casting them.
//
// The field is WORLD anchored, not view anchored. The old dome was dir.xz / dir.y, which
// paints clouds onto the sky as a function of where you look — they never move as you walk
// and have no position to cast a shadow from. Anchoring costs a camera-position uniform and
// buys parallax, flying through them, and #32 for nearly free.

const float kCloudBase = 420.0; // above kTerrainTop (384), so terrain never intersects
const float kCloudTop = 560.0;
const float kCloudMid = 0.5 * (kCloudBase + kCloudTop);

float cl_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float cl_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(cl_hash(i), cl_hash(i + vec2(1, 0)), f.x),
               mix(cl_hash(i + vec2(0, 1)), cl_hash(i + vec2(1, 1)), f.x), f.y);
}

float cl_fbm_n(vec2 p, int octaves) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < octaves; ++i) {
        s += a * cl_noise(p);
        p = p * 2.02 + 3.1;
        a *= 0.5;
    }
    // Rescaled to the FIVE-octave range, not to 1.0. Every threshold in this file was tuned
    // against five octaves, and a three-octave sum only reaches 0.875 of the same total — so
    // an unscaled cheap field is a thinner one, and the clouds and their shadows would stop
    // agreeing about where they are. Anchoring to five rather than normalising to one is
    // what keeps the five-octave path returning exactly what it always did.
    const float kFiveOctaveSum = 1.0 - 0.03125; // 1 - 0.5^5
    return s * (kFiveOctaveSum / (1.0 - pow(0.5, float(octaves))));
}

float cl_fbm(vec2 p) { return cl_fbm_n(p, 5); }

vec2 cl_domain(vec2 worldXz, float time) {
    return worldXz * 0.0016 + vec2(time * 0.0045, time * 0.0028);
}

// 0..1 coverage at a world XZ. Wind drifts the field rather than the camera, so standing
// still still shows movement.
float cloudCoverage(vec2 worldXz, float time) {
    return cl_fbm_n(cl_domain(worldXz, time), 5);
}

// The same field at three octaves, for the terrain's shadow lookup only.
//
// The two finest octaves land at roughly 78 and 39 blocks. A cloud shadow is averaged across
// the thickness of the layer, which smears it by tens to hundreds of blocks — so that detail
// cannot survive the averaging and is being computed only to be blurred away. Dropping it
// pays for the extra taps, and the taps are what actually fix the banding.
float cloudCoverageShadow(vec2 worldXz, float time) {
    return cl_fbm_n(cl_domain(worldXz, time), 3);
}

// Density inside the slab. The height profile is what makes it a volume rather than a
// sheet: thin at the base, fullest around the middle, feathered at the top, so a raymarch
// through it produces soft edges instead of a hard lid.
float cloudDensity(vec3 worldPos, float time) {
    if (worldPos.y < kCloudBase || worldPos.y > kCloudTop)
        return 0.0;
    float cov = cloudCoverage(worldPos.xz, time);
    float d = smoothstep(0.46, 0.70, cov);
    if (d <= 0.0)
        return 0.0;
    float h = (worldPos.y - kCloudBase) / (kCloudTop - kCloudBase); // 0..1 up the slab
    float profile = smoothstep(0.0, 0.22, h) * (1.0 - smoothstep(0.55, 1.0, h));
    return d * profile;
}

// Taps taken through the THICKNESS of the layer when shadowing terrain.
//
// One tap treats the clouds as a plane, and a plane has no penumbra: coverage crosses the
// threshold over a very short distance on the ground, so the shadow arrives as a hard edge
// and the only softness left is whatever gradient the fbm happened to have at that point.
// That is #58b — broad diagonal bands with crisp edges sweeping the terrain.
//
// Successive taps are offset from one another by the ray's OWN travel through the slab, so
// nothing has to be invented to spread them: 140 blocks of thickness works out to roughly
// 440 blocks of lateral spread at a low sun and 70 at a high one. The smear therefore widens
// exactly when the banding is worst, which is also what a real cloud does.
// Three octaves each rather than five (see cloudCoverageShadow), so High costs 9 noise
// lookups against the old 5 — not 25.
//
// Both includers set this at their own top, but GLSL's preprocessor treats an undefined name
// in #if as 0 — so a third includer that forgot would silently get the ONE-tap path and the
// banding back, with nothing to indicate why. Defaulting here matches chunk.frag and
// composite.frag rather than relying on them.
#ifndef VX_QUALITY
#define VX_QUALITY 2
#endif
#if VX_QUALITY <= 0
#define VX_CLOUD_SHADOW_TAPS 1
#elif VX_QUALITY == 1
#define VX_CLOUD_SHADOW_TAPS 2
#elif VX_QUALITY == 2
#define VX_CLOUD_SHADOW_TAPS 3
#else
#define VX_CLOUD_SHADOW_TAPS 5
#endif

// How much sunlight reaches worldPos through the cloud layer, 1 = clear. Used by the terrain
// shader. Still far cheaper than the sky's raymarch — it is a fixed handful of 2D fbm
// lookups, not a march.
float cloudSunlight(vec3 worldPos, vec3 sunDir, float time) {
    if (sunDir.y <= 0.05)
        return 1.0; // sun at or below the horizon: no meaningful cloud shadow
    if (worldPos.y >= kCloudBase)
        return 1.0; // inside or above the layer already

    float shade = 0.0;
    float weight = 0.0;
    for (int i = 0; i < VX_CLOUD_SHADOW_TAPS; ++i) {
        // Spread across the slab, avoiding the very edges where the density profile has
        // feathered to nothing anyway.
        float f = (VX_CLOUD_SHADOW_TAPS == 1)
                      ? 0.5
                      : mix(0.15, 0.85, float(i) / float(VX_CLOUD_SHADOW_TAPS - 1));
        float h = mix(kCloudBase, kCloudTop, f);
        float t = (h - worldPos.y) / sunDir.y;
        vec2 hit = worldPos.xz + sunDir.xz * t;
        // The same height profile cloudDensity uses, as the tap weight — otherwise the thin
        // feathered base and top count as much as the solid middle and the shadow reads
        // heavier than the cloud looks.
        float w = smoothstep(0.0, 0.22, f) * (1.0 - smoothstep(0.55, 1.0, f)) + 0.08;
        // A wider toe than the sky's own 0.46..0.70. A shadow edge as crisp as the cloud
        // edge reads as a stencil, and a cloud this far up has an enormous penumbra.
        shade += w * smoothstep(0.38, 0.88, cloudCoverageShadow(hit, time));
        weight += w;
    }
    return 1.0 - 0.65 * (shade / weight);
}
