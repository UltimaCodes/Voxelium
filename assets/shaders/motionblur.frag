#version 460 core

// camera motion blur. reproject each pixel's world position into the previous frame (same
// camera-relative reconstruction as TAA) to get its screen-space velocity, then average a
// few taps along that velocity. only the camera moves the world here, so this is a
// per-pixel velocity from depth, no velocity buffer needed.

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uColor; // resolved HDR
layout(binding = 1) uniform sampler2D uDepth;

layout(location = 0) uniform vec2 uProjScale;
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot;
layout(location = 5) uniform mat4 uPrevViewProj;
layout(location = 9) uniform vec3 uCamDelta;
layout(location = 10) uniform float uStrength;

void main() {
    float rawD = texture(uDepth, vUv).r;
    if (rawD <= 0.0) { // sky: leave it (world velocity is ill-defined at infinity)
        outColor = texture(uColor, vUv);
        return;
    }
    vec2 ndc = vUv * 2.0 - 1.0;
    float viewZ = -uNear / max(rawD, 1e-7);
    vec3 worldRel = uInvViewRot *
                    vec3(ndc.x * uProjScale.x * -viewZ, ndc.y * uProjScale.y * -viewZ, viewZ);
    vec4 pc = uPrevViewProj * vec4(worldRel + uCamDelta, 1.0);
    vec2 prevUv = (pc.xy / pc.w) * 0.5 + 0.5;

    vec2 vel = (vUv - prevUv) * uStrength;
    float vl = length(vel);
    if (vl > 0.06) // clamp so a big jump doesn't smear the whole screen
        vel *= 0.06 / vl;

    const int N = 8;
    vec3 sum = vec3(0.0);
    for (int i = 0; i < N; ++i) {
        vec2 uv = vUv - vel * (float(i) / float(N - 1) - 0.5);
        sum += texture(uColor, clamp(uv, 0.0, 1.0)).rgb;
    }
    outColor = vec4(sum / float(N), 1.0);
}
