#version 460 core

// bloom mix, exposure, ACES filmic (Narkowicz fit), manual sRGB encode — manual so the
// output is identical on every driver instead of trusting GL_FRAMEBUFFER_SRGB support

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uHdr;
layout(binding = 2) uniform sampler2D uBloom;
layout(location = 0) uniform float uBloomStrength;
layout(location = 1) uniform float uExposure;

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(uHdr, vUv).rgb;
    hdr += texture(uBloom, vUv).rgb * uBloomStrength; // additive: bloom is pre-thresholded
    vec3 mapped = aces(hdr * uExposure);
    outColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
