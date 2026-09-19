#version 460 core

// Every creature is a procedural shape in sprite space (x along the heading, -1..1 both ways).
// No texture: a butterfly is two ellipses whose width follows the flap, a fish is an ellipse and
// a triangle, and so on. Cheap, resolution independent, and animated for free from the phase.

in vec2 vP;
in vec4 vTint;
flat in vec2 vKindPhase;
layout(location = 0) out vec4 outColor;

const int kButterfly = 1, kFirefly = 2, kGull = 3, kFish = 4, kLeaf = 5, kMote = 6, kBubble = 7,
          kBird = 8;

void main() {
    int kind = int(vKindPhase.x + 0.5);
    float phase = vKindPhase.y;
    vec3 rgb = vTint.rgb;
    float a = 0.0;
    vec2 p = vP;

    if (kind == kButterfly) {
        // Wings are two lobes either side of a thin body; the flap squeezes them in x, and the
        // front pair is bigger than the back pair.
        float flap = abs(cos(phase));
        vec2 q = vec2(abs(p.x) / max(flap, 0.12), p.y);
        float front = length((q - vec2(0.42, 0.34)) * vec2(1.0, 0.85));
        float back = length((q - vec2(0.36, -0.30)) * vec2(1.15, 1.0));
        float wing = max(step(front, 0.5), step(back, 0.36));
        float body = step(abs(p.x), 0.06) * step(abs(p.y), 0.55);
        // a dark rim and a pale spot so it reads as a wing, not a flat blob
        float rim = smoothstep(0.42, 0.5, front) + smoothstep(0.28, 0.36, back);
        rgb = mix(rgb, rgb * 0.35, clamp(rim, 0.0, 1.0) * wing);
        rgb = mix(rgb, vec3(1.0), 0.6 * step(length(q - vec2(0.42, 0.42)), 0.09) * wing);
        rgb = mix(rgb, vec3(0.08), body);
        a = max(wing, body);
    } else if (kind == kFirefly) {
        float d = length(p);
        float pulse = 0.55 + 0.45 * sin(phase);
        // a hot core inside a soft halo; the HDR tint is what makes the bloom pick it up
        a = (exp(-d * d * 7.0) * 0.9 + step(d, 0.18) * 0.6) * pulse;
        rgb *= 0.6 + 1.6 * exp(-d * d * 14.0);
    } else if (kind == kGull || kind == kBird) {
        // seen from below: a shallow W, the wingtips lagging behind the body as they beat
        float flap = sin(phase) * 0.35;
        float y = 0.10 - 0.36 * abs(p.x) - flap * abs(p.x) * abs(p.x);
        float d = abs(p.y - y);
        float w = 0.13 * (1.0 - 0.7 * abs(p.x));
        a = smoothstep(w, w * 0.4, d) * step(abs(p.x), 0.96);
        a = max(a, step(length(p * vec2(1.0, 1.9)), 0.16)); // the body
    } else if (kind == kFish) {
        // heading is +x: body ellipse, then a tail that wags
        float wag = sin(phase) * 0.28;
        float body = step(length((p - vec2(-0.12, 0.0)) / vec2(0.62, 0.30)), 1.0);
        float tx = (p.x - 0.42) / 0.5;
        float tail = step(0.0, tx) * step(tx, 1.0) * step(abs(p.y - wag * tx), 0.06 + 0.42 * tx);
        a = max(body, tail);
        float belly = smoothstep(-0.05, 0.28, -p.y);
        rgb = mix(rgb, rgb * 1.5 + 0.12, belly * body);
        rgb = mix(rgb, rgb * 0.55, smoothstep(0.02, 0.24, p.y) * body); // darker back
        rgb = mix(rgb, vec3(0.03), step(length(p - vec2(-0.55, 0.06)), 0.055)); // eye
        rgb = mix(rgb, rgb * 0.6, 0.5 * step(abs(fract(p.x * 3.2 + 0.5) - 0.5), 0.09) * body); // a stripe
    } else if (kind == kLeaf) {
        // a pointed oval that tumbles: squeezed by the phase so it flips as it falls
        float sq = 0.35 + 0.65 * abs(cos(phase));
        vec2 q = vec2(p.x, p.y / sq);
        a = step(abs(q.y), 0.55 * (1.0 - abs(q.x) * 0.9)) * step(abs(q.x), 1.0);
        rgb *= 0.8 + 0.4 * abs(p.y);
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

    // Airborne creatures are cutouts that WRITE DEPTH (the C++ side draws them in their own
    // batch): composite paints the sky over any pixel whose depth is still the far plane, so a
    // soft-blended gull against the sky was erased outright. With depth, the composite fogs it
    // like the terrain at its distance, which is also the right look. Fish and glows stay
    // soft and depth-free - a fish writing depth would read to the water pass as a patch of
    // shallow water and grow foam around itself.
    bool hard = kind == kButterfly || kind == kGull || kind == kLeaf || kind == kBird;
    if (hard) {
        if (a < 0.5)
            discard;
        outColor = vec4(rgb, vTint.a);
        return;
    }
    if (a <= 0.02)
        discard;
    outColor = vec4(rgb, a * vTint.a);
}
