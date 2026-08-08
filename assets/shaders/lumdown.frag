#version 460 core

// Scene luminance into a small square target, in LOG space. Log rather than linear because
// the average is then a geometric mean: one specular highlight or a lamp filling a few
// pixels cannot drag the whole exposure down the way an arithmetic mean lets it.
//
// The caller mips this down to 1x1, which is the scene average the adaptation pass chases.

in vec2 vUv;
layout(location = 0) out float outLum;

layout(binding = 0) uniform sampler2D uHdr;
layout(location = 0) uniform vec2 uTexel; // source texel, for the spread of the taps

void main() {
    // Four bilinear taps rather than one: each output texel covers ~20x11 source pixels at
    // 1280x720, so a single point sample is a lottery over one of them and the exposure
    // visibly shimmers as the camera moves.
    vec2 o = uTexel * 2.0;
    vec3 c = texture(uHdr, vUv + vec2(-o.x, -o.y)).rgb +
             texture(uHdr, vUv + vec2(o.x, -o.y)).rgb +
             texture(uHdr, vUv + vec2(-o.x, o.y)).rgb +
             texture(uHdr, vUv + vec2(o.x, o.y)).rgb;
    float l = dot(c * 0.25, vec3(0.2126, 0.7152, 0.0722));
    outLum = log(max(l, 1e-4));
}
