#version 460 core

layout(location = 5) uniform sampler2DArray uBlockTex;
layout(location = 6) uniform float uLayer; // crack stage layer in the block texture array

in vec2 vUv;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 c = texture(uBlockTex, vec3(vUv, uLayer));
    if (c.a < 0.02)
        discard; // the untouched part of the tile must leave the block showing
    outColor = c;
}
