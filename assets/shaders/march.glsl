// Shared by composite.frag and scatter.frag. The includer declares uShadowMap, uCascadeVp,
// uCascadeSplit and VX_SHADOW_RES before including this.

// 4-tap softened shadow lookup for the raymarch, a single tap makes the light shafts
// alias into a grid against the shadow texels; averaging a few taps smooths them out.
float shadowSingle(vec3 pos) {
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    vec4 sc = uCascadeVp[c] * vec4(pos, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float z = sc.z - 0.0015;
    float t = 1.5 / VX_SHADOW_RES;
    return 0.25 * (texture(uShadowMap, vec4(uv + vec2(t, t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(-t, t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(t, -t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(-t, -t), float(c), z)));
}

// interleaved gradient noise, a much better dither than a plain hash, so the raymarch
// start offset breaks up banding evenly instead of in blotches
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
