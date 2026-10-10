#version 460 core

// one screen-space quad, positioned/sized in pixels (origin bottom-left). used for every
// hotbar element, panel, selection frame, block icons.
layout(location = 0) uniform vec4 uRectPx; // (left, bottom, width, height) in pixels
layout(location = 1) uniform vec2 uInvRes; // (1/width, 1/height)

out vec2 vUv;

const vec2 kC[6] =
    vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));

void main() {
    vec2 c = kC[gl_VertexID];
    vec2 px = uRectPx.xy + c * uRectPx.zw;
    vUv = vec2(c.x, 1.0 - c.y); // texture top-left origin so icons aren't upside down
    gl_Position = vec4(px * uInvRes * 2.0 - 1.0, 0.0, 1.0);
}
