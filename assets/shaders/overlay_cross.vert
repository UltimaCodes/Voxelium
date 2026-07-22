#version 460 core

// two thin bars forming a centre crosshair, sized in pixels. two triangles per bar,
// vertex-pulled from gl_VertexID.
layout(location = 0) uniform vec2 uInvRes; // (1/width, 1/height)

const vec2 kUnit[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1),
                              vec2(1, 1), vec2(-1, 1));

void main() {
    int bar = gl_VertexID / 6;
    vec2 c = kUnit[gl_VertexID % 6];
    vec2 halfPx = bar == 0 ? vec2(10.0, 1.0) : vec2(1.0, 10.0); // horizontal, vertical
    gl_Position = vec4(c * halfPx * uInvRes * 2.0, 0.0, 1.0);
}
