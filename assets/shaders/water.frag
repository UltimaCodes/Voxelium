#version 460 core

// screen-space-reflection water. reflects the opaque scene (trees, terrain) by ray
// marching the reflection ray against the depth buffer and sampling a snapshot of the
// opaque colour; falls back to a computed sky where the ray leaves the screen. fresnel
// drives both the reflection strength and the alpha, so looking down you see through to
// the seabed (the blend leaves the opaque pixel behind) and at grazing angles you see
// the mirror. animated wave normal, sharp sun specular.

in vec3 vWorldRel; // camera-relative world position
flat in uint vDir;
flat in float vSky;

layout(location = 0) out vec4 outColor;

layout(location = 0) uniform mat4 uViewProj; // occupies 0..3 (same as the vertex stage)
layout(location = 4) uniform float uTime;
layout(location = 5) uniform vec3 uSunL;
layout(location = 6) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 7) uniform float uNear;
layout(location = 8) uniform vec3 uSunColor; // day/night sun radiance

layout(binding = 2) uniform sampler2D uSceneColor; // opaque snapshot
layout(binding = 3) uniform sampler2D uSceneDepth;

const vec3 kNormal[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));

// sum of four directional waves -> organic surface. returns the xz slope gradient so the
// normal varies smoothly and irregularly (a single sine product makes a lattice of
// specular dots — that was the "squares on the water").
vec2 waveGrad(vec2 p) {
    const vec4 W[4] = vec4[4](vec4(0.80, 0.20, 0.28, 0.055), vec4(-0.30, 0.95, 0.41, 0.040),
                              vec4(0.55, -0.72, 0.67, 0.024), vec4(-0.92, -0.35, 0.93, 0.015));
    const float sp[4] = float[4](1.1, 1.5, 2.1, 2.7);
    vec2 g = vec2(0.0);
    for (int i = 0; i < 4; ++i) {
        vec2 d = normalize(W[i].xy);
        float ph = dot(d, p) * W[i].z + uTime * sp[i];
        g += d * W[i].z * W[i].w * cos(ph);
    }
    return g;
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

// march R from world-rel point P, return reflected colour (scene hit or sky)
vec3 reflection(vec3 P, vec3 R) {
    float stepLen = 0.6;
    vec3 sp = P;
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
        g = waveGrad(vWorldRel.xz);
        N = normalize(vec3(-g.x * 6.0, 1.0, -g.y * 6.0));
    }
    vec3 V = normalize(-vWorldRel); // camera at the origin
    float fres = mix(0.02, 1.0, pow(1.0 - max(dot(N, V), 0.0), 5.0));

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

    // Beer-Lambert absorption: shallow shows the bed, deep goes blue-green
    float absorb = 1.0 - exp(-thickness * 0.18);
    vec3 tint = mix(vec3(0.12, 0.30, 0.36), vec3(0.02, 0.09, 0.16), absorb);
    refr = mix(refr, tint, absorb) * (0.4 + 0.6 * vSky);

    vec3 refl = reflection(vWorldRel, reflect(-V, N)) * (0.4 + 0.6 * vSky);

    vec3 H = normalize(uSunL + V);
    float spec = pow(max(dot(N, H), 0.0), 120.0) * 1.1 * vSky;

    vec3 col = mix(refr, refl, fres) + uSunColor * 0.5 * spec; // glint fades at night

    // Shoreline foam: thin water gets an animated fringe. It has to be LIT like everything
    // else - as a constant white it stayed full brightness after dark, so every shoreline
    // glowed against black water, which is the "water is luminescent at night" and the
    // white edge wherever water meets a block. Tie it to the sky light reaching this
    // surface and to how much sun there is, so at night it goes to the same near-black the
    // rest of the water does.
    float shore = 1.0 - smoothstep(0.0, 1.6, thickness);
    float fn = fbm2(vWorldRel.xz * 1.3 + vec2(uTime * 0.3, -uTime * 0.2));
    float foam = smoothstep(0.55, 0.95, shore * (0.55 + 0.6 * fn));
    float daylight = clamp(uSunL.y * 3.0 + 0.12, 0.0, 1.0);
    vec3 foamCol = vec3(0.92, 0.96, 1.0) * vSky * (0.10 + 0.90 * daylight);
    col = mix(col, foamCol, foam);

    outColor = vec4(col, 1.0); // fully composited here (refraction sampled directly)
}
