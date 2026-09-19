#version 460 core

in vec3 vColor;
in float vAlpha;
layout(location = 0) out vec4 outColor;

float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

void main() {
    // Solid geometry cannot be alpha blended without ordering problems between its own parts,
    // so a creature fading in or out at the edge of its ring is dithered away instead.
    if (vAlpha < 0.98 && ign(gl_FragCoord.xy) > vAlpha)
        discard;
    outColor = vec4(vColor, 1.0);
}
