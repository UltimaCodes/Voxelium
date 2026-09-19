#version 460 core

// Lit and coloured, nothing more: no shadows, no texture, no detail. It is only ever seen
// through haze, and the composite pass melts it into the sky with the same distance fog as
// everything else.
in vec3 vColor;
in vec3 vNormal;
in vec3 vRel;
flat in float vWater;
layout(location = 0) out vec4 outColor;

layout(location = 6) uniform float uInner;
layout(location = 7) uniform vec3 uSunL;
layout(location = 8) uniform vec3 uSunColor;
layout(location = 9) uniform vec3 uSkyAmbient;

void main() {
    // the streamed chunks own everything inside this radius
    if (length(vRel.xz) < uInner - 24.0)
        discard;
    vec3 N = normalize(vNormal);
    float ndl = max(dot(N, uSunL), 0.0);
    vec3 col = vColor * (uSkyAmbient * 1.15 + uSunColor * ndl * 0.85);
    if (vWater > 0.5) {
        vec3 V = normalize(-vRel);
        float glint = pow(max(dot(reflect(-uSunL, vec3(0.0, 1.0, 0.0)), V), 0.0), 48.0);
        col += uSunColor * glint * 0.6;
        float fres = pow(1.0 - clamp(V.y, 0.0, 1.0), 3.0);
        col = mix(col, uSkyAmbient * 1.6, 0.35 * fres);
    }
    outColor = vec4(col, 1.0);
}
