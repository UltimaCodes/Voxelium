#version 460 core

// applies SSAO to the scene and adds volumetric sun scattering raymarched through the
// shadow cascades. output stays HDR; bloom and tonemapping come after.

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uScene;
layout(binding = 2) uniform sampler2D uAo;
layout(binding = 3) uniform sampler2D uDepth;
layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;

layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot; // view -> camera-relative world
layout(location = 3) uniform float uAoOn;
layout(location = 4) uniform float uVolumetricOn;
layout(location = 5) uniform vec3 uSunL;
layout(location = 6) uniform mat4 uCascadeVp[3];
layout(location = 18) uniform vec3 uCascadeSplit;

float shadowSingle(vec3 pos) {
    float d = length(pos);
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    if (d >= uCascadeSplit.z)
        return 1.0;
    vec4 sc = uCascadeVp[c] * vec4(pos, 1.0);
    return texture(uShadowMap, vec4(sc.xy * 0.5 + 0.5, float(c), sc.z - 0.0015));
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec3 color = texture(uScene, vUv).rgb;
    if (uAoOn > 0.5)
        color *= texture(uAo, vUv).r;

    if (uVolumetricOn > 0.5) {
        float rawD = texture(uDepth, vUv).r;
        float viewZ = -uNear / max(rawD, 1e-7);
        vec2 ndc = vUv * 2.0 - 1.0;
        vec3 vpos = vec3(ndc.x * uProjScale.x * -viewZ, ndc.y * uProjScale.y * -viewZ, viewZ);
        vec3 world = uInvViewRot * vpos;
        float len = min(length(world), 140.0);
        vec3 dir = normalize(world);

        const int kSteps = 12;
        float stepLen = len / float(kSteps);
        float t = stepLen * hash(gl_FragCoord.xy); // jitter hides banding
        float scatter = 0.0;
        for (int i = 0; i < kSteps; ++i) {
            scatter += shadowSingle(dir * t);
            t += stepLen;
        }
        scatter /= float(kSteps);
        // forward-scattering phase: subtle everywhere, strong looking into the sun
        float phase = 0.10 + 0.90 * pow(max(dot(dir, uSunL), 0.0), 8.0);
        color += vec3(1.15, 1.05, 0.88) * (scatter * phase * len * 0.0007);
    }

    outColor = vec4(color, 1.0);
}
