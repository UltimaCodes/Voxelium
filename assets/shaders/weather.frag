#version 460 core

in float vFade;
in vec2 vUv;
in float vTone;
layout(location = 0) out vec4 outColor;

layout(location = 10) uniform float uStrength; // 0..1, how hard it is coming down
// Same location as the vertex stage: a uniform of one name must sit at one location across
// the whole program, and declaring it twice at different numbers is a link error.
layout(location = 8) uniform float uSnow; // 0 rain .. 1 snow
layout(location = 11) uniform vec3 uTint; // sky-lit, so precipitation goes dark at night

void main() {
    // Rain is a thin ribbon: bright at the head, thinning to nothing up the tail, with a soft
    // lateral falloff. Snow is a round flake. The per-drop tone is what stops a whole sheet of
    // rain reading as one streak stamped over and over.
    //
    // Both bases are clamped. Interpolation leaves the edge pixels of a streak a hair past 0 or
    // 1, pow() of a negative base is NaN on this driver, NaN passes the discard below, and bloom
    // spreads that one pixel into a black square a third of the screen wide for a frame, the
    // owner's black squares in the rain, about one every two seconds in a downpour.
    float lateral = abs(vUv.x * 2.0 - 1.0);
    float streak = pow(clamp(1.0 - lateral, 0.0, 1.0), 1.6) * pow(clamp(1.0 - vUv.y, 0.0, 1.0), 0.8) *
                   smoothstep(0.0, 0.06, vUv.y + 0.02);
    float flake = 1.0 - smoothstep(0.25, 0.5, length(vUv - 0.5));
    float shape = mix(streak, flake, uSnow);

    // Deliberately faint. Rain that is clearly visible per-drop reads as debris; what sells
    // it is a lot of barely-there streaks, and the alpha is what keeps that from turning
    // into a grey wash over the whole frame.
    float alpha = shape * vFade * uStrength * mix(0.55, 0.75, uSnow) * mix(vTone, 1.0, uSnow);
    if (alpha < 0.004)
        discard;
    outColor = vec4(uTint, alpha);
}
