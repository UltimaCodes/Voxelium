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

vec3 skyColor(vec3 d) {
    float up = clamp(d.y, 0.0, 1.0);
    vec3 col = mix(vec3(0.66, 0.78, 0.92), vec3(0.14, 0.34, 0.80), pow(up, 0.5));
    float sd = max(dot(d, uSunL), 0.0);
    col += vec3(1.0, 0.55, 0.28) * pow(sd, 8.0) * 0.6;
    col += vec3(1.0, 0.95, 0.85) * smoothstep(0.9985, 0.9995, sd) * 9.0;
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
    if (vDir == 2u) { // organic ripple on the top surface
        vec2 g = waveGrad(vWorldRel.xz) * 6.0;
        N = normalize(vec3(-g.x, 1.0, -g.y));
    }
    vec3 V = normalize(-vWorldRel); // camera at the origin
    float fres = mix(0.02, 1.0, pow(1.0 - max(dot(N, V), 0.0), 5.0));

    // how much water sits between the surface and the seabed at this pixel -> clear
    // shallows, deep opaque blue
    vec2 suv = gl_FragCoord.xy / vec2(textureSize(uSceneDepth, 0));
    float bedZ = -uNear / max(texture(uSceneDepth, suv).r, 1e-7);
    float surfZ = -(uViewProj * vec4(vWorldRel, 1.0)).w;
    float thickness = max(surfZ - bedZ, 0.0);
    float deep = 1.0 - exp(-thickness * 0.11);

    vec3 tint = mix(vec3(0.10, 0.30, 0.38), vec3(0.01, 0.055, 0.12), deep) * (0.3 + 0.7 * vSky);
    vec3 refl = reflection(vWorldRel, reflect(-V, N)) * (0.35 + 0.65 * vSky);

    // broader, softer sun glint (was a razor-thin pow(400) that only lit the lattice)
    vec3 H = normalize(uSunL + V);
    float spec = pow(max(dot(N, H), 0.0), 90.0) * 0.9 * vSky;

    vec3 col = mix(tint, refl, fres) + vec3(1.0, 0.96, 0.86) * spec;
    // shallow + steep -> low alpha (see the seabed); deep or grazing -> opaque water
    float alpha = clamp(mix(mix(0.35, 0.9, deep), 1.0, fres) + spec, 0.0, 1.0);
    outColor = vec4(col, alpha);
}
