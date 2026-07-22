#version 460 core

// 9-tap tent upsample; blended additively onto the next mip up

in vec2 vUv;
layout(location = 0) out vec3 outColor;
layout(binding = 0) uniform sampler2D uSrc;

void main() {
    vec2 ts = 1.0 / vec2(textureSize(uSrc, 0));
    vec3 sum = texture(uSrc, vUv + ts * vec2(-1, 1)).rgb * 1.0 +
               texture(uSrc, vUv + ts * vec2(0, 1)).rgb * 2.0 +
               texture(uSrc, vUv + ts * vec2(1, 1)).rgb * 1.0 +
               texture(uSrc, vUv + ts * vec2(-1, 0)).rgb * 2.0 +
               texture(uSrc, vUv).rgb * 4.0 +
               texture(uSrc, vUv + ts * vec2(1, 0)).rgb * 2.0 +
               texture(uSrc, vUv + ts * vec2(-1, -1)).rgb * 1.0 +
               texture(uSrc, vUv + ts * vec2(0, -1)).rgb * 2.0 +
               texture(uSrc, vUv + ts * vec2(1, -1)).rgb * 1.0;
    outColor = sum / 16.0;
}
