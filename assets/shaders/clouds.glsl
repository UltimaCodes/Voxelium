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

float cl_fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; ++i) {
        s += a * cl_noise(p);
        p = p * 2.02 + 3.1;
        a *= 0.5;
    }
    return s;
}

// 0..1 coverage at a world XZ. Wind drifts the field rather than the camera, so standing
// still still shows movement.
float cloudCoverage(vec2 worldXz, float time) {
    vec2 p = worldXz * 0.0016 + vec2(time * 0.0045, time * 0.0028);
    return cl_fbm(p);
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

// How much sunlight reaches worldPos through the cloud layer, 1 = clear. Used by the
// terrain shader; a single coverage lookup where the sun ray crosses the middle of the
// slab, which is the cheap approximation that still tracks the clouds you can see.
float cloudSunlight(vec3 worldPos, vec3 sunDir, float time) {
    if (sunDir.y <= 0.05)
        return 1.0; // sun at or below the horizon: no meaningful cloud shadow
    float t = (kCloudMid - worldPos.y) / sunDir.y;
    if (t <= 0.0)
        return 1.0; // above the layer already
    vec2 hit = worldPos.xz + sunDir.xz * t;
    float cov = cloudCoverage(hit, time);
    // Softer threshold than the render: a shadow edge as crisp as the cloud edge reads as a
    // stencil, and the penumbra of a cloud 200 blocks up is enormous anyway.
    return 1.0 - 0.65 * smoothstep(0.44, 0.78, cov);
}
