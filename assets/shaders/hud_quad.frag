#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(location = 2) uniform vec4 uColor; // solid fill, or icon tint
layout(location = 3) uniform int uLayer;  // < 0 => solid uColor (panels, water)
layout(binding = 0) uniform sampler2DArray uTex;

void main() {
    outColor = uLayer < 0 ? uColor : texture(uTex, vec3(vUv, float(uLayer))) * uColor;
}
