#version 460 core

in float vFade;
in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(location = 10) uniform float uStrength; // 0..1, how hard it is coming down
// Same location as the vertex stage: a uniform of one name must sit at one location across
// the whole program, and declaring it twice at different numbers is a link error.
layout(location = 8) uniform float uSnow; // 0 rain .. 1 snow
layout(location = 11) uniform vec3 uTint; // sky-lit, so precipitation goes dark at night

void main() {
    // Soften the ends of a streak and round a flake, so neither reads as a hard rectangle.
    float a = 1.0 - abs(vUv.y * 2.0 - 1.0);
    float flake = 1.0 - smoothstep(0.25, 0.5, length(vUv - 0.5));
    float shape = mix(a * a, flake, uSnow);

    // Deliberately faint. Rain that is clearly visible per-drop reads as debris; what sells
    // it is a lot of barely-there streaks, and the alpha is what keeps that from turning
    // into a grey wash over the whole frame.
    float alpha = shape * vFade * uStrength * mix(0.30, 0.75, uSnow);
    if (alpha < 0.004)
        discard;
    outColor = vec4(uTint, alpha);
}
