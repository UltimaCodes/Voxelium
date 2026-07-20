#version 460 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;

// Explicit uniform locations — the engine never does glGetUniformLocation string lookups.
layout(location = 0) uniform float uTime;
layout(location = 1) uniform float uAspect;

out vec3 vColor;

void main() {
    float c = cos(uTime);
    float s = sin(uTime);
    vec2 p = mat2(c, s, -s, c) * aPos; // GLSL mat2 is column-major: this spins CCW
    p.x /= uAspect;                    // undo window aspect so the triangle isn't stretched
    gl_Position = vec4(p, 0.0, 1.0);
    vColor = aColor;
}
