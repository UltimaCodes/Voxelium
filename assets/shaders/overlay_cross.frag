#version 460 core

layout(location = 0) out vec4 outColor;

// drawn with GL_ONE_MINUS_DST_COLOR so this inverts whatever's behind it — readable on
// any background, no matter the scene colour.
void main() { outColor = vec4(1.0); }
