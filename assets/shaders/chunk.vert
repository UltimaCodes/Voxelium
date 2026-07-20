#version 460 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor; // face shade baked in at mesh time

layout(location = 0) uniform mat4 uViewProj;

out vec3 vColor;

void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    vColor = aColor.rgb;
}
