#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(location = 2) uniform vec4 uColor; // solid fill, or icon tint
layout(location = 3) uniform int uLayer;  // < 0 => solid uColor (panels, water)
layout(binding = 0) uniform sampler2DArray uTex;

void main() {
    if (uLayer < 0) {
        outColor = uColor;
        return;
    }
    // block tiles are sRGB storage and read back linear, this draws straight to the screen
    vec4 t = texture(uTex, vec3(vUv, float(uLayer)));
    outColor = vec4(pow(t.rgb, vec3(1.0 / 2.2)), t.a) * uColor;
}
