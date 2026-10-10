#version 460 core

// Scene luminance into a small square target, in LOG space. Log rather than linear because
// the average is then a geometric mean: one specular highlight or a lamp filling a few
// pixels cannot drag the whole exposure down the way an arithmetic mean lets it.
//
// Metered on the world, not the sky. A frame that is a third bright sky averaged high enough
// to darken everything under it, and from any height the dark greens of the canopies went near
// black while the grass beside them still read as lit, as if the light never reached the
// treetops. So r is the log luminance of the geometry only (weighted by g, the share of the
// taps that hit any), and b the whole frame's, which takes over when there is nothing but sky.
//
// The caller mips this down to 1x1, which is the scene average the adaptation pass chases.

in vec2 vUv;
layout(location = 0) out vec4 outLum;

layout(binding = 0) uniform sampler2D uHdr;
layout(binding = 1) uniform sampler2D uDepth; // reversed-Z: 0 where nothing was drawn
layout(location = 0) uniform vec2 uTexel; // source texel, for the spread of the taps

void main() {
    // Four taps rather than one: each output texel covers ~20x11 source pixels at 1280x720, so
    // a single point sample is a lottery over one of them and the exposure visibly shimmers as
    // the camera moves.
    vec2 o = uTexel * 2.0;
    const vec2 taps[4] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    float world = 0.0, weight = 0.0, frame = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec2 uv = vUv + taps[i] * o;
        float l = log(max(dot(texture(uHdr, uv).rgb, vec3(0.2126, 0.7152, 0.0722)), 1e-4));
        float w = texture(uDepth, uv).r > 0.0 ? 1.0 : 0.0;
        world += w * l;
        weight += w;
        frame += l;
    }
    outLum = vec4(world, weight, frame, 0.0) * 0.25;
}
