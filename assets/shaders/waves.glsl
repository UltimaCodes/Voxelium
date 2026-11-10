// The swell, shared by water.tese (the surface it moves) and water.frag (the normal it shades
// with). The parameters come from render/WaterWaves.cpp, which also evaluates the same sum on the
// CPU for the camera's underwater test.
layout(location = 26) uniform vec4 uWave[4];    // direction xz, wavenumber (radians a block), weight
layout(location = 30) uniform vec4 uWavePhase;  // each wave's phase at the camera, time included
layout(location = 31) uniform float uWaveScale; // blocks at full strength, weather included; 0 = flat
layout(location = 32) uniform vec2 uWaveFade;   // distance the swell starts / finishes fading out
layout(location = 33) uniform vec4 uWaveShape;  // s, exp(-2s), mean of exp(s(sin - 1)), 2 / (1 - exp(-2s))
layout(location = 36) uniform vec4 uGroup;      // phases at the camera: two wave groups, the shore's two wanders
layout(location = 37) uniform float uBreakClock; // breakers come ashore one a unit of this

// Waves run in sets: two slow beats travelling with the swell, so a stretch of sea builds and dies.
// The same numbers as WaterWaves.cpp, which evaluates all this for the camera's underwater test.
float waveGroup(vec2 rel) {
    float a = sin(0.0419 * dot(uWave[0].xy, rel) + uGroup.x);
    float b = sin(0.0648 * dot(uWave[2].xy, rel) + uGroup.y);
    return 0.70 + 0.32 * a + 0.20 * b;
}

// How far out to sea a camera-relative point is, in blocks, from the generator's own coast map, so
// crests follow the shape of the shore and not the bumps of the sea floor. < 0 inland. Large where
// there is no map.
layout(binding = 9) uniform sampler2D uCoast;
layout(location = 38) uniform vec3 uCoastXf; // corner relative to the camera, 1 / span; z = 0 for no map

float seaDistance(vec2 rel) {
    return uCoastXf.z > 0.0 ? -textureLod(uCoast, (rel - uCoastXf.xy) * uCoastXf.z, 0.0).r : 1e4;
}

// A breaker d blocks out to sea. x is its height at a swell scale of 1; y how much has broken into foam.
vec2 breaker(vec2 rel, float d) {
    if (d > 80.0 || d < -6.0)
        return vec2(0.0);
    float wander = 0.6 * sin(dot(vec2(0.031, 0.017), rel) + uGroup.z) + 0.4 * sin(dot(vec2(-0.023, 0.041), rel) + uGroup.w);
    float psi = d * 0.058 + uBreakClock + wander * 0.26;
    float idx = floor(psi + 0.5);
    float u = psi - idx; // 0 on the crest, negative shoreward
    // steep in front, a long back
    float crest = u < 0.0 ? exp(-u * u * 90.0) : exp(-u * u * 16.0);
    // every wave its own size, and each one uneven along its length
    float size = 0.40 + 0.60 * (0.5 + 0.5 * sin(idx * 1.7 + wander * 2.2));
    // builds coming in, tallest just before it breaks, spent by the waterline
    float shoal = smoothstep(80.0, 26.0, d) * (1.0 + 0.9 * smoothstep(34.0, 12.0, d)) * smoothstep(-2.0, 7.0, d);
    float broken = smoothstep(17.0, 8.0, d) * size;
    float foam = broken * max(crest, smoothstep(-0.5, -0.04, u) * 0.8 * (1.0 - smoothstep(-0.04, 0.02, u)));
    return vec2(1.6 * shoal * size * crest, foam);
}

// Height (x) and gradient (yz) of the unit swell at a camera-relative point. Each wave is
// exp(s(sin - 1)) rescaled to average zero: a sharp crest and a broad, flat trough, which is the
// shape of real swell rather than a plain sine's even ripple.
vec3 swell(vec2 rel) {
    float h = 0.0;
    vec2 g = vec2(0.0);
    float group = waveGroup(rel);
    for (int i = 0; i < 4; ++i) {
        float th = uWave[i].z * dot(uWave[i].xy, rel) + uWavePhase[i];
        float e = exp(uWaveShape.x * (sin(th) - 1.0));
        float a = uWave[i].w * uWaveShape.w;
        h += a * (e - uWaveShape.z);
        g += (a * e * uWaveShape.x * cos(th) * uWave[i].z) * uWave[i].xy;
    }
    return vec3(h, g) * group;
}

float waveFade(float d) {
    return 1.0 - smoothstep(uWaveFade.x, uWaveFade.y, d);
}
