// The sky's colour in a direction: deep overhead, bright at the horizon, warm toward a low sun. No
// disc, stars or clouds. Shared by the sky, the fog that melts into it and the water that reflects it.
vec3 skyGradient(vec3 d, vec3 sunL, float day, float overcast) {
    float up = max(d.y, 0.0);
    vec3 zenith = mix(vec3(0.004, 0.007, 0.022), vec3(0.050, 0.155, 0.58), day);
    vec3 middle = mix(vec3(0.009, 0.015, 0.038), vec3(0.17, 0.38, 0.88), day);
    vec3 horizon = mix(vec3(0.020, 0.030, 0.055), vec3(0.60, 0.80, 1.10), day);
    vec3 col = mix(horizon, middle, 1.0 - exp(-up * 7.0));
    col = mix(col, zenith, smoothstep(0.12, 0.85, up));
    if (d.y < 0.0)
        col = mix(horizon, mix(vec3(0.016, 0.020, 0.030), vec3(0.30, 0.37, 0.48), day), clamp(-d.y * 3.0, 0.0, 1.0));

    float alt = sunL.y;
    // a low sun: blue draining out of the sky, a hot band along the horizon under it, rose above that.
    // Some of it lasts into the morning and from the late afternoon.
    float low = smoothstep(0.55, 0.04, abs(alt)) * smoothstep(-0.20, -0.02, alt);
    vec2 flat2 = d.xz / max(length(d.xz), 1e-4);
    vec2 sunFlat = sunL.xz / max(length(sunL.xz), 1e-4);
    float toward = 0.5 + 0.5 * dot(flat2, sunFlat);
    col = mix(col, col * vec3(0.78, 0.70, 0.86), low * 0.6);
    col += vec3(1.9, 0.72, 0.20) * (exp(-up * 10.0) * toward * toward * low * 1.5);
    col += vec3(0.95, 0.44, 0.42) * (exp(-up * 3.6) * (0.28 + 0.72 * toward) * low * 0.42);

    // the glow around the sun itself, and the wide one the haze scatters forward, which is what makes
    // the land toward a low sun read backlit and hazy (the fog takes its colour from here)
    float sd = max(dot(d, sunL), 0.0);
    float above = smoothstep(-0.06, 0.04, alt);
    col += vec3(1.0, 0.82, 0.62) * (pow(sd, 32.0) * 0.30 + pow(sd, 220.0) * 0.8) * above * mix(0.5, 1.0, low);
    vec3 glow = mix(vec3(1.0, 0.58, 0.28), vec3(1.0, 0.86, 0.68), smoothstep(0.06, 0.5, alt));
    col += glow * (pow(sd, 6.0) * 0.20 + sd * sd * 0.05) * above * smoothstep(0.0, 0.12, day) *
           mix(0.45, 1.3, low) * (1.0 - 0.5 * up);

    // rain: the colour goes out of it and the whole sky drops to a flat grey
    float grey = dot(col, vec3(0.30, 0.55, 0.15));
    return mix(col, vec3(grey) * vec3(0.72, 0.76, 0.82), overcast * 0.85);
}
