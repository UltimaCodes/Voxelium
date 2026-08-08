#version 460 core

// Eye adaptation: one texel holding the log luminance the exposure is currently chasing.
// Ping-ponged, so this reads last frame's value and writes this frame's — the whole state
// of the effect is one float, and it never touches the CPU (a readback would stall the
// frame to save nothing).

layout(location = 0) out float outAdapt;

layout(binding = 0) uniform sampler2D uLum;  // mipped log-luminance; last mip is the average
layout(binding = 1) uniform sampler2D uPrev; // 1x1, last frame's adapted value

layout(location = 0) uniform float uLastMip;
layout(location = 1) uniform float uRateBright; // 1 - exp(-dt/tau), walking into light
layout(location = 2) uniform float uRateDark;   // ditto, into darkness
layout(location = 3) uniform float uValid;      // 0 = first frame, snap instead of blending

void main() {
    float target = textureLod(uLum, vec2(0.5), uLastMip).r;
    float prev = texture(uPrev, vec2(0.5)).r;
    if (uValid < 0.5) {
        outAdapt = target; // snapping beats fading up from black on the first frame
        return;
    }
    // Asymmetric on purpose, and this is the difference between "adapts" and "feels like
    // eyes": squinting into brightness settles quickly, coming into a dark cave takes
    // noticeably longer.
    float rate = target > prev ? uRateBright : uRateDark;
    outAdapt = mix(prev, target, clamp(rate, 0.0, 1.0));
}
