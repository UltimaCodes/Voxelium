#version 460 core

// 4x4 box blur, enough to hide the SSAO rotation noise at half res

in vec2 vUv;
layout(location = 0) out float outV;
layout(binding = 0) uniform sampler2D uSrc;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(uSrc, 0));
    float sum = 0.0;
    for (int y = -1; y <= 2; ++y)
        for (int x = -1; x <= 2; ++x)
            sum += texture(uSrc, vUv + (vec2(x, y) - 0.5) * texel).r;
    outV = sum / 16.0;
}
