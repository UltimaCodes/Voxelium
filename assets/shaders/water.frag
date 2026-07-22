#version 460 core

// stylized water: fresnel-tinted deep/shallow blend, a sharp sun specular for the
// glinty highlight, sky-light dimming underground, alpha-blended over the opaque scene.
// HDR out (blended before tonemapping). animated normal from the same wave field.

in vec3 vWorldRel;
flat in uint vDir;
flat in float vSky;

layout(location = 0) out vec4 outColor;

layout(location = 4) uniform vec3 uSunL;
layout(location = 1) uniform float uTime;

const vec3 kNormal[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));

void main() {
    vec3 N = kNormal[vDir];
    if (vDir == 2u) { // perturb the top-surface normal with the wave gradient
        float dx = 0.10 * cos(vWorldRel.x * 0.7 + uTime * 1.6);
        float dz = 0.10 * cos(vWorldRel.z * 0.9 + uTime * 1.3);
        N = normalize(vec3(-dx, 1.0, -dz));
    }
    vec3 V = normalize(-vWorldRel); // camera sits at the origin (camera-relative space)
    float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);

    vec3 deep = vec3(0.015, 0.07, 0.13);
    vec3 shallow = vec3(0.08, 0.26, 0.36);
    vec3 base = mix(shallow, deep, 0.55) * (0.25 + 0.75 * vSky);

    vec3 H = normalize(uSunL + V);
    float spec = pow(max(dot(N, H), 0.0), 200.0) * 2.2 * vSky;

    vec3 col = base + vec3(0.55, 0.68, 0.85) * (fres * 0.5 * vSky) + vec3(1.0, 0.96, 0.86) * spec;
    float alpha = mix(0.72, 0.95, fres);
    outColor = vec4(col, alpha);
}
