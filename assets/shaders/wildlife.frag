#version 460 core

// The glows: each is a soft procedural shape in sprite space. Everything with a body is real
// geometry now (wildlife3d.vert).

in vec2 vP;
in vec4 vTint;
flat in vec2 vKindPhase;
layout(location = 0) out vec4 outColor;

const int kFirefly = 2, kMote = 6, kBubble = 7;

void main() {
    int kind = int(vKindPhase.x + 0.5);
    float phase = vKindPhase.y;
    vec3 rgb = vTint.rgb;
    float a = 0.0;
    vec2 p = vP;

    if (kind == kFirefly) {
        float d = length(p);
        float pulse = 0.55 + 0.45 * sin(phase);
        // a hot core inside a soft halo; the HDR tint is what makes the bloom pick it up
        a = (exp(-d * d * 7.0) * 0.9 + step(d, 0.18) * 0.6) * pulse;
        rgb *= 0.6 + 1.6 * exp(-d * d * 14.0);
    } else if (kind == kBubble) {
        float d = length(p);
        float ring = smoothstep(0.62, 0.86, d) * step(d, 1.0);
        float glint = step(length(p - vec2(-0.3, 0.3)), 0.16);
        a = max(ring * 0.75, glint);
        rgb = mix(rgb, vec3(1.0), glint);
    } else { // mote: a soft dot
        float d = length(p);
        a = exp(-d * d * 3.5) * 0.85;
    }
    if (a <= 0.02)
        discard;
    outColor = vec4(rgb, a * vTint.a);
}
