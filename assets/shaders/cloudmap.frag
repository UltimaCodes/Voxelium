#version 460 core

// Cloud coverage as a small world-anchored texture, so terrain reads a cloud shadow with one
// filtered fetch instead of evaluating the fbm per fragment (that cost 1-3.5 ms of a 720p
// scene). The field only depends on world XZ and time, never on the sun or the receiver, so
// it can be rendered once per frame and sampled at whatever point the sun ray crosses the
// cloud layer. chunk.frag does the sampling.

#ifndef VX_QUALITY
#define VX_QUALITY 2
#endif

in vec2 vUv;
layout(location = 0) out float outShade;

layout(location = 0) uniform vec2 uOrigin; // world XZ of the texture's lower-left corner
layout(location = 1) uniform float uSpan;  // world size of the whole texture
layout(location = 2) uniform float uTime;

#include "clouds.glsl"

void main() {
    vec2 xz = uOrigin + vUv * uSpan;
    outShade = smoothstep(0.15 - uCloudBias * 0.6, 1.05 - uCloudBias * 0.6,
                          cloudCoverageShadow(xz, uTime));
}
