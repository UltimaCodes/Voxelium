#version 460 core

// Cloud cover as a small world-anchored texture, so terrain reads a cloud shadow with one filtered
// fetch. It only depends on world XZ and time, never on the sun or the receiver, so it is rendered
// once a frame and sampled at whatever point the sun ray crosses the cloud layer.

in vec2 vUv;
layout(location = 0) out float outShade;

layout(location = 0) uniform vec2 uOrigin; // world XZ of the texture's lower-left corner
layout(location = 1) uniform float uSpan;  // world size of the whole texture
layout(location = 2) uniform float uTime;

#include "clouds.glsl"
#include "cloudfield.glsl"

void main() {
    outShade = cloudCoverage(cloudWeather(uOrigin + vUv * uSpan, uTime).r);
}
