// The swell, shared by water.tese (the surface it moves) and water.frag (the normal it shades
// with). The parameters come from render/WaterWaves.cpp, which also evaluates the same sum on the
// CPU for the camera's underwater test.
layout(location = 26) uniform vec4 uWave[4];    // direction xz, wavenumber (radians a block), weight
layout(location = 30) uniform vec4 uWavePhase;  // each wave's phase at the camera, time included
layout(location = 31) uniform float uWaveScale; // blocks at full strength, weather included; 0 = flat
layout(location = 32) uniform vec2 uWaveFade;   // distance the swell starts / finishes fading out
layout(location = 33) uniform vec4 uWaveShape;  // s, exp(-2s), mean of exp(s(sin - 1)), 2 / (1 - exp(-2s))

// Height (x) and gradient (yz) of the unit swell at a camera-relative point. Each wave is
// exp(s(sin - 1)) rescaled to average zero: a sharp crest and a broad, flat trough, which is the
// shape of real swell rather than a plain sine's even ripple.
vec3 swell(vec2 rel) {
    float h = 0.0;
    vec2 g = vec2(0.0);
    for (int i = 0; i < 4; ++i) {
        float th = uWave[i].z * dot(uWave[i].xy, rel) + uWavePhase[i];
        float e = exp(uWaveShape.x * (sin(th) - 1.0));
        float a = uWave[i].w * uWaveShape.w;
        h += a * (e - uWaveShape.z);
        g += (a * e * uWaveShape.x * cos(th) * uWave[i].z) * uWave[i].xy;
    }
    return vec3(h, g);
}

float waveFade(float d) {
    return 1.0 - smoothstep(uWaveFade.x, uWaveFade.y, d);
}
