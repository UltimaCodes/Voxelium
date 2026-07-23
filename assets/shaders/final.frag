#version 460 core

// bloom mix, exposure, ACES filmic (Narkowicz fit), manual sRGB encode — manual so the
// output is identical on every driver instead of trusting GL_FRAMEBUFFER_SRGB support.
// then an optional unsharp pass on the tonemapped image to claw back the softness TAA
// leaves (done in LDR so bright HDR highlights don't ring).

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uHdr;
layout(binding = 2) uniform sampler2D uBloom;
layout(location = 0) uniform float uBloomStrength;
layout(location = 1) uniform float uExposure;
layout(location = 3) uniform float uSharpen; // 0 = off
layout(location = 4) uniform vec2 uTexel;    // 1/resolution

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 tonemap(vec3 hdr) { return pow(aces(hdr * uExposure), vec3(1.0 / 2.2)); }

void main() {
    vec3 hdr = texture(uHdr, vUv).rgb + texture(uBloom, vUv).rgb * uBloomStrength;
    vec3 c = tonemap(hdr);

    if (uSharpen > 0.0) {
        vec3 l = tonemap(texture(uHdr, vUv - vec2(uTexel.x, 0.0)).rgb);
        vec3 r = tonemap(texture(uHdr, vUv + vec2(uTexel.x, 0.0)).rgb);
        vec3 u = tonemap(texture(uHdr, vUv - vec2(0.0, uTexel.y)).rgb);
        vec3 d = tonemap(texture(uHdr, vUv + vec2(0.0, uTexel.y)).rgb);
        c += (c * 4.0 - l - r - u - d) * uSharpen; // unsharp mask
    }
    outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
