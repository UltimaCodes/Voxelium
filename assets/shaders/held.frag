#version 460 core

layout(binding = 0) uniform sampler2DArray uBlockTex;
layout(location = 4) uniform vec3 uLight; // the light where you stand, already in display terms

in vec3 vUvl;
in float vShade;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 t = texture(uBlockTex, vUvl);
    if (t.a < 0.5)
        discard;
    // the array is sRGB, read back linear: lit there, then back to display for the finished frame
    vec3 lit = t.rgb * vShade * uLight;
    outColor = vec4(pow(lit, vec3(1.0 / 2.2)), 1.0);
}
