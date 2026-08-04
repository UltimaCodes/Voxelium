#version 460 core

layout(location = 1) uniform vec4 uColor;

layout(location = 0) out vec4 outColor;

// Flat colour, picked in the settings menu. This used to draw with GL_ONE_MINUS_DST_COLOR
// to invert whatever was behind it, which reads on any background but costs a driver
// shader patch (dst-colour blending into an sRGB target).
void main() { outColor = uColor; }
