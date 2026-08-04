#version 460 core

// applies SSAO to the scene, paints a real sky for pixels with no geometry (gradient +
// sun disc + horizon glow — no extra pass), and adds volumetric sun scattering
// raymarched through the shadow cascades for geometry pixels. output stays HDR; bloom
// and tonemapping come after.

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uScene;
layout(binding = 2) uniform sampler2D uAo;
layout(binding = 3) uniform sampler2D uDepth;
layout(binding = 4) uniform sampler2D uSsgi;
layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;

layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot; // view -> camera-relative world; eats 2,3,4
layout(location = 5) uniform vec3 uSunL;
layout(location = 6) uniform mat4 uCascadeVp[3];
layout(location = 18) uniform vec3 uCascadeSplit;
layout(location = 19) uniform float uUnderwater; // 1 when the camera eye is submerged
layout(location = 20) uniform float uCloudsOn;
layout(location = 21) uniform float uTime;
layout(location = 22) uniform float uSsgiOn;
layout(location = 23) uniform float uAoOn;         // moved off 3 (was aliased by the mat3)
layout(location = 24) uniform float uVolumetricOn; // moved off 4
layout(location = 25) uniform vec3 uSunColor;      // day/night sun radiance
layout(location = 26) uniform vec2 uFog;           // (start, end) distance; huge = off

const vec3 kWaterDeep = vec3(0.03, 0.13, 0.20);

vec3 viewRay() {
    vec2 ndc = vUv * 2.0 - 1.0;
    return normalize(uInvViewRot * vec3(ndc.x * uProjScale.x, ndc.y * uProjScale.y, -1.0));
}

float starHash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// Cube-face parameterisation of a direction: pick the dominant axis and divide by it, so
// every face gets an evenly spaced 2D grid. Folding a 3D direction into 2D any other way
// (e.g. dir.xy + dir.z) makes whole families of directions share a cell, and the stars
// come out as streaks instead of points.
vec2 cubeFaceUv(vec3 dir, out float face) {
    vec3 a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) {
        face = dir.x > 0.0 ? 0.0 : 1.0;
        return dir.zy / a.x;
    }
    if (a.y >= a.z) {
        face = dir.y > 0.0 ? 2.0 : 3.0;
        return dir.xz / a.y;
    }
    face = dir.z > 0.0 ? 4.0 : 5.0;
    return dir.xy / a.z;
}

// One star per grid cell, placed at a random point inside it and drawn as a round dot
// with a soft edge. Most cells are empty; brightness and colour vary so the field reads
// as a sky rather than a regular dot screen.
vec3 starField(vec3 dir, float night) {
    float face;
    vec2 uv = cubeFaceUv(dir, face) * 90.0;
    vec2 cell = floor(uv);
    vec2 f = uv - cell;
    vec2 key = cell + face * 311.0;

    float pick = starHash(key);
    if (pick < 0.960)
        return vec3(0.0);

    vec2 centre = vec2(starHash(key + 11.3), starHash(key + 23.7)) * 0.7 + 0.15;
    float d = length(f - centre);
    float mag = starHash(key + 41.1);            // most stars faint, a few bright
    float radius = mix(0.055, 0.13, mag * mag);
    float dot0 = smoothstep(radius, 0.0, d);
    // slow, per-star twinkle
    float tw = 0.72 + 0.28 * sin(uTime * 1.7 + pick * 90.0);
    // colour: cool white through faint gold
    vec3 tint = mix(vec3(0.78, 0.85, 1.0), vec3(1.0, 0.92, 0.78), starHash(key + 57.9));
    return tint * dot0 * tw * (0.55 + 1.35 * mag * mag) * night;
}

// smooth value noise — the blocky floor() cells of a raw hash were visible as squares
// across something as small on screen as the moon
float vnoise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(starHash(i), starHash(i + vec2(1.0, 0.0)), f.x),
               mix(starHash(i + vec2(0.0, 1.0)), starHash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// A moon that reads as a sphere, not a white disc: cratered surface, curvature shading
// toward the limb, and a soft halo. Kept deliberately dim in HDR — pushed much above 1
// the tonemapper crushes the whole surface to flat white, which is what made it look
// like a small sun.
vec3 moonDisc(vec3 dir, vec3 moonDir, float night) {
    const float kRadius = 0.030; // angular radius
    float md = dot(dir, moonDir);
    if (md <= 0.0)
        return vec3(0.0);

    vec3 mx = normalize(cross(moonDir, vec3(0.0, 1.0, 0.0)) + vec3(1e-4, 0.0, 0.0));
    vec3 my = cross(mx, moonDir);
    vec2 mc = vec2(dot(dir, mx), dot(dir, my)) / kRadius; // -1..1 across the face
    float r = length(mc);

    vec3 halo = vec3(0.22, 0.26, 0.38) * pow(max(md, 0.0), 1400.0) * night * 0.7;
    if (r >= 1.0)
        return halo;

    float limb = sqrt(max(1.0 - r * r, 0.0)); // sphere normal's z: darkens toward the edge
    // maria + craters, a few octaves over the face
    float m = vnoise2(mc * 2.1 + 3.0) * 0.55 + vnoise2(mc * 5.3 + 11.0) * 0.30 +
              vnoise2(mc * 12.0 + 23.0) * 0.15;
    vec3 surface =
        mix(vec3(0.34, 0.36, 0.42), vec3(0.88, 0.88, 0.83), smoothstep(0.30, 0.78, m));
    float edge = smoothstep(1.0, 0.93, r); // antialias the silhouette
    return (surface * (0.45 + 0.55 * limb) * night * edge) + halo;
}

vec3 skyColor(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    float alt = uSunL.y;                             // sun altitude, -1..1
    float day = clamp(alt * 3.0 + 0.12, 0.0, 1.0);   // 1 day, 0 night
    // day/night gradient
    vec3 zenith = mix(vec3(0.01, 0.015, 0.05), vec3(0.14, 0.34, 0.80), day);
    vec3 horizon = mix(vec3(0.04, 0.06, 0.12), vec3(0.66, 0.78, 0.92), day);
    vec3 col = mix(horizon, zenith, pow(up, 0.5));
    if (dir.y < 0.0)
        col = mix(horizon, mix(vec3(0.06, 0.07, 0.09), vec3(0.34, 0.38, 0.44), day),
                  clamp(-dir.y * 3.0, 0.0, 1.0));

    float sd = max(dot(dir, uSunL), 0.0);
    float above = smoothstep(-0.03, 0.06, alt);
    float lowSun = clamp(1.0 - abs(alt) * 2.2, 0.0, 1.0) * day; // sunrise/sunset warmth
    col += vec3(1.0, 0.34, 0.10) * pow(sd, 3.0) * lowSun * 2.4;            // warm band
    col = mix(col, col * vec3(1.2, 0.92, 0.72), lowSun * 0.5);            // golden-hour cast
    col += vec3(1.0, 0.55, 0.28) * pow(sd, 8.0) * 0.6 * above;             // glow
    col += uSunColor * smoothstep(0.9986, 0.9995, sd) * 3.2 * above;       // sun disc

    // Stars fade out well before the sky is properly light — a linear 1-day left them
    // visible as bright specks through dawn and dusk.
    float night = pow(1.0 - day, 2.2);
    if (night > 0.004) {
        col += starField(dir, night) * step(-0.02, dir.y); // none below the horizon
        col += moonDisc(dir, -uSunL, night);               // moon rides opposite the sun
    }
    return col;
}

// drifting fbm cloud dome: flatten the view direction onto a plane (perspective toward the
// horizon), fbm coverage, and a cheap sun-side density difference for shading + silver
// lining. smooth noise -> soft organic clouds, never blocky.
float vhash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(vhash(i), vhash(i + vec2(1, 0)), f.x),
               mix(vhash(i + vec2(0, 1)), vhash(i + vec2(1, 1)), f.x), f.y);
}
float cloudFbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; ++i) {
        s += a * vnoise(p);
        p = p * 2.02 + 3.1;
        a *= 0.5;
    }
    return s;
}
vec4 clouds(vec3 dir) {
    if (dir.y < 0.03)
        return vec4(0.0);
    vec2 p = dir.xz / dir.y * 0.6 + vec2(uTime * 0.008, uTime * 0.005); // flatten + wind
    float cov = cloudFbm(p);
    float d = smoothstep(0.42, 0.66, cov) * smoothstep(0.03, 0.22, dir.y);
    if (d <= 0.001)
        return vec4(0.0);
    vec2 sp = (uSunL.xz / max(abs(uSunL.y), 0.25)) * 0.12; // one step toward the sun
    float shade = clamp((cov - cloudFbm(p + sp)) * 2.2 + 0.55, 0.25, 1.0);
    vec3 col = mix(vec3(0.52, 0.56, 0.64), vec3(1.05, 1.02, 0.98), shade);
    float day = clamp(uSunL.y * 3.0 + 0.12, 0.0, 1.0);
    col += vec3(1.0, 0.55, 0.2) * pow(max(dot(dir, uSunL), 0.0), 6.0) * day; // warm sun side
    col *= 0.16 + 0.84 * day; // clouds go dim + cool at night (moonlit, not sunlit)
    return vec4(col, d);
}
vec3 skyWithClouds(vec3 dir) {
    vec3 s = skyColor(dir);
    if (uCloudsOn > 0.5) {
        vec4 c = clouds(dir);
        s = mix(s, c.rgb, c.a);
    }
    return s;
}

// 4-tap softened shadow lookup for the raymarch — a single tap makes the light shafts
// alias into a grid against the shadow texels; averaging a few taps smooths them out.
float shadowSingle(vec3 pos) {
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    vec4 sc = uCascadeVp[c] * vec4(pos, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float z = sc.z - 0.0015;
    float t = 1.5 / 1536.0;
    return 0.25 * (texture(uShadowMap, vec4(uv + vec2(t, t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(-t, t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(t, -t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(-t, -t), float(c), z)));
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// interleaved gradient noise — a much better dither than a plain hash, so the raymarch
// start offset breaks up banding evenly instead of in blotches
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

void main() {
    float rawD = texture(uDepth, vUv).r;
    vec3 dir = viewRay();

    // sky pixel (reversed-z far = 0): paint the sky instead of the flat clear colour.
    // underwater there is no sky — the far distance is just deep water.
    if (rawD <= 0.0) {
        outColor = vec4(uUnderwater > 0.5 ? kWaterDeep : skyWithClouds(dir), 1.0);
        return;
    }

    vec3 color = texture(uScene, vUv).rgb;
    if (uAoOn > 0.5)
        color *= texture(uAo, vUv).r;
    if (uSsgiOn > 0.5)
        color += texture(uSsgi, vUv).rgb * 0.9; // one-bounce indirect fill

    // reconstruct the camera-relative world position once, for volumetrics + fog
    float viewZ = -uNear / max(rawD, 1e-7);
    vec2 ndc = vUv * 2.0 - 1.0;
    vec3 worldRel = uInvViewRot *
                    vec3(ndc.x * uProjScale.x * -viewZ, ndc.y * uProjScale.y * -viewZ, viewZ);
    float dist = length(worldRel);

    if (uVolumetricOn > 0.5) {
        float len = min(dist, 150.0);
        vec3 rd = normalize(worldRel);

        const int kSteps = 16;
        float stepLen = len / float(kSteps);
        float t = stepLen * ign(gl_FragCoord.xy); // dithered start, breaks up banding
        float scatter = 0.0;
        for (int i = 0; i < kSteps; ++i) {
            scatter += shadowSingle(rd * t);
            t += stepLen;
        }
        scatter /= float(kSteps);
        // stronger forward lobe + a touch more intensity so the god-ray shafts read
        // clearly when looking toward the sun, without bringing the aliased grid back
        float phase = 0.10 + 0.90 * pow(max(dot(rd, uSunL), 0.0), 5.0);
        // uSunColor carries the day/night intensity, so the shafts fade out at night
        color += uSunColor * (scatter * phase * len * 0.00018);
    }

    if (uUnderwater > 0.5) {
        // underwater: absorb reds and fog toward deep water with distance (Beer-Lambert-ish)
        color *= vec3(0.55, 0.82, 1.0);
        color = mix(color, kWaterDeep, 1.0 - exp(-dist * 0.10));
    } else {
        // distance fog: melt the far edge of the loaded world into the sky, so the render
        // boundary reads as haze/horizon instead of a hard cut
        float fog = smoothstep(uFog.x, uFog.y, dist);
        if (fog > 0.001)
            color = mix(color, skyWithClouds(dir), fog);
    }

    outColor = vec4(color, 1.0);
}
