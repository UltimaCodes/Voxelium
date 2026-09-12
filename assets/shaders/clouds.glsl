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

    // Owner report, 2026-09-12: hard, regular diagonal checkering on terrain AND water,
    // sweeping/moving as time of day changes. Confirmed by direct A/B
    // (VX_TOUR_CLOUDSHADOW=0.0 vs 0.6, identical camera) to be this term, not the AO gating
    // (#58, already fixed this session — always-on day or night, so it could not be this) and
    // not shadow-map acne (#68-followup, already fixed — a real, separate bug, but that fix
    // barely moved this symptom, see SPEC). The multi-tap averaging added for #58b measurably
    // made banding WORSE, not better (SPEC #58b-followup: 0.1521 -> 0.1795 band-hardness),
    // and now we know why: every tap lands on the SAME LINE (the sun azimuth through
    // worldPos, offset only by height fraction f) through a quasi-periodic fbm field, at a
    // FIXED relative spacing that is a direct function of sun elevation (see the
    // lateral-spread comment above) — that is exactly "moves with time of day". Averaging
    // colinear samples of a periodic-ish signal at fixed spacing is a comb filter, not a
    // blur: at the wrong spacing relative to the field's own octave wavelengths it beats
    // instead of smoothing. A first attempt jittering each tap's height fraction by a few
    // percent measured as barely doing anything (SPEC): the dominant octave here has a
    // ~625-block wavelength, so a jitter small enough not to swap tap order never moved any
    // sample more than a few percent of that wavelength either.
    //
    // Fixed properly: each tap gets an independent 2D offset PERPENDICULAR to the sun azimuth
    // (30-70 blocks, comparable to the finest surviving octave's own ~150-block wavelength —
    // see cloudCoverageShadow), not just jittered along the ray. That turns "N colinear points"
    // into "N points scattered over a real 2D neighbourhood", which is an actual area low-pass
    // on the noise field and cannot resonate the way a comb can, regardless of sun angle. The
    // per-pixel hash still means a single static frame shows fine dither rather than fully
    // smoothed — TAA resolves that the same way it does for #68-followup and ssgi.frag's
    // kernel. The coverage->shadow toe is also widened well past #58b's 0.38..0.88, so a
    // threshold crossing is a long, gentle ramp instead of a stencil edge no matter how well
    // blurred the input is.
    vec2 sunAz = normalize(sunDir.xz + vec2(1e-6)); // avoid NaN when the sun is straight up
    vec2 perp = vec2(-sunAz.y, sunAz.x);

    // The 2D jitter above still wasn't the full story: even a 70-block scatter barely dents
    // the DOMINANT octave (~625-block wavelength, by far the largest-amplitude term in the
    // 3-octave sum), and the checkering in the owner's screenshot is visibly only a few
    // blocks across -- far finer than even the finest surviving shadow octave (~150 blocks,
    // see cloudCoverageShadow's own comment). A pattern finer than the noise field's own
    // detail cannot BE the noise field's detail; it's aliasing. On a grazing or receding
    // surface (a distant terrace riser, water running out to the horizon), one screen pixel's
    // footprint in worldPos.xz grows large from ordinary perspective compression, so a
    // fixed-frequency field gets undersampled into fine fake structure -- the same reason an
    // unmipped distant checkerboard floor shimmers in any renderer. No amount of
    // same-frequency point-sampling fixes that; the fix is the same one texture mipmapping
    // uses: fade toward a flat, untextured answer once the sample footprint approaches the
    // field's own feature size, rather than confidently aliasing. Measured with fwidth() at
    // the slab's own mid-height (one derivative pair, not per-tap -- cheap, and terrain
    // fragments never reach the divergent early-return above since kCloudBase is already
    // above kTerrainTop, so this derivative is always taken in uniform control flow for the
    // caller that actually matters here).
    vec2 midHit = worldPos.xz + sunDir.xz * ((kCloudMid - worldPos.y) / sunDir.y);
    float aliasFade = smoothstep(40.0, 160.0, length(fwidth(midHit)));
    if (aliasFade >= 0.999)
        return 1.0;

    // The actual dominant cause, found after the fixes above measured as barely moving this
    // view (SPEC): a stepped/terraced receiver's worldPos.y jumps by whole blocks between
    // adjacent, visually near-identical steps, and t = (h - worldPos.y) / sunDir.y divides
    // that jump by sunDir.y — at a low sun this turns one block of ordinary stair-step height
    // into a large lateral jump in `hit`, so neighbouring steps sample noticeably different
    // points of the field even though nothing about the CLOUD changed between them. That is a
    // real geometric effect, not a screen-space one, which is why neither the area jitter
    // above nor the fwidth-based alias fade (both keyed on how `hit` varies per PIXEL) touched
    // it: the jump happens between adjacent geometry, not within a pixel's own footprint, and
    // it reproduces exactly the reported behaviour — worse at a lower sun (bigger 1/sunDir.y),
    // hence "moves as time of day changes" is really "gets worse or better", not literal
    // motion of a fixed pattern. Fixed by rounding the height used for this lookup to a coarse
    // 24-block grid: a run of stair steps within the same bracket now shares one lateral
    // sample instead of jittering block to block, and the field's own large wavelength (>=150
    // blocks, see cloudCoverageShadow) means a shadow does not need per-block height
    // resolution to look right. The 24-block bracket boundaries this introduces are rare
    // compared to per-block steps and get the same per-pixel jitter as everything else below.
    float shadowY = round(worldPos.y / 24.0) * 24.0;

    float shade = 0.0;
    float weight = 0.0;
    for (int i = 0; i < VX_CLOUD_SHADOW_TAPS; ++i) {
        // Spread across the slab, avoiding the very edges where the density profile has
        // feathered to nothing anyway.
        float f = (VX_CLOUD_SHADOW_TAPS == 1)
                      ? 0.5
                      : mix(0.15, 0.85, float(i) / float(VX_CLOUD_SHADOW_TAPS - 1));
        float h = mix(kCloudBase, kCloudTop, f);
        float t = (h - shadowY) / sunDir.y;
        vec2 hit = worldPos.xz + sunDir.xz * t;

        vec2 seed = gl_FragCoord.xy + vec2(float(i) * 23.0, float(i) * 59.0);
        float jAngle = fract(sin(dot(seed, vec2(19.81, 74.23))) * 31415.9) * 6.2831853;
        float jDist = 30.0 + 40.0 * fract(sin(dot(seed, vec2(41.7, 8.23))) * 27183.1);
        hit += perp * (cos(jAngle) * jDist) + sunAz * (sin(jAngle) * jDist * 0.3);

        // The same height profile cloudDensity uses, as the tap weight — otherwise the thin
        // feathered base and top count as much as the solid middle and the shadow reads
        // heavier than the cloud looks.
        float w = smoothstep(0.0, 0.22, f) * (1.0 - smoothstep(0.55, 1.0, f)) + 0.08;
        // Toe widened from #58b's 0.38..0.88 to most of the field's own range. Upper end past
        // 1.0: the 3-octave rescale (see cloudCoverageShadow) can overshoot the true
        // 5-octave range slightly.
        shade += w * smoothstep(0.15, 1.05, cloudCoverageShadow(hit, time));
        weight += w;
    }
    float lit = 1.0 - 0.6 * (shade / weight);
    return mix(lit, 1.0, aliasFade);
}
