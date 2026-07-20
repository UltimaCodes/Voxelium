#version 460 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;

// Explicit uniform locations — the engine never does glGetUniformLocation string lookups.
layout(location = 0) uniform float uTime;
layout(location = 1) uniform mat4 uViewProj; // mat4 consumes locations 1..4

out vec3 vColor;

void main() {
    float c = cos(uTime);
    float s = sin(uTime);
    vec3 p = vec3(aPos.x * c, aPos.y, aPos.x * s); // spin the triangle about the Y axis
    gl_Position = uViewProj * vec4(p, 1.0);
    vColor = aColor;
}
