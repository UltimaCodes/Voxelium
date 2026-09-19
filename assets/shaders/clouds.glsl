// Shared cloud field. Included by composite.frag (which raymarches it for the sky) and by
// cloudmap.frag (which bakes the coverage into a texture the terrain reads its cloud shadows
// from). One copy, because a second one would drift and the symptom would be shadows that do
// not match the clouds casting them.
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
    // Lower threshold nudged from 0.46 (#79): reference "hazy mountain" shots read as more
    // cloud-present overall, not clear sky with occasional puffs - widening the low end
    // means more of the coverage field's own range counts as cloud, without touching the
    // 0.70 upper end that already governs how solid a fully-covered patch reads.
    float d = smoothstep(0.40, 0.70, cov);
    if (d <= 0.0)
        return 0.0;
    float h = (worldPos.y - kCloudBase) / (kCloudTop - kCloudBase); // 0..1 up the slab
    float profile = smoothstep(0.0, 0.22, h) * (1.0 - smoothstep(0.55, 1.0, h));
    return d * profile;
}

// Terrain cloud shadows do not evaluate this file per fragment any more: cloudmap.frag renders
// cloudCoverageShadow into a small world-anchored texture once a frame and chunk.frag reads it
// back along the sun ray. Same field, so a shadow still belongs to a cloud you can see, at a
// fraction of the cost and without the per-pixel jitter the old multi-tap version needed to
// hide its comb-filter banding.
