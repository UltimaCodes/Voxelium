#version 460 core

// 13-tap downsample (Jimenez, "Next Generation Post Processing in Call of Duty").
// center 2x2 block gets half the weight, the four overlapping corner blocks share the
// rest — kills the fireflies a naive box chain produces.

in vec2 vUv;
layout(location = 0) out vec3 outColor;
layout(binding = 0) uniform sampler2D uSrc;

// only >0 on the first downsample: keeps bloom to genuinely bright pixels (sun-lit
// highlights, glow blocks) instead of hazing the whole frame
layout(location = 0) uniform float uThreshold;
layout(location = 1) uniform float uKnee;

vec3 prefilter(vec3 c) {
    if (uThreshold <= 0.0)
        return c;
    float b = max(c.r, max(c.g, c.b));
    float soft = clamp(b - uThreshold + uKnee, 0.0, 2.0 * uKnee);
    soft = soft * soft / (4.0 * uKnee + 1e-5);
    float contrib = max(soft, b - uThreshold) / max(b, 1e-5);
    return c * contrib;
}

void main() {
    vec2 ts = 1.0 / vec2(textureSize(uSrc, 0));
    vec3 a = texture(uSrc, vUv + ts * vec2(-2, 2)).rgb;
    vec3 b = texture(uSrc, vUv + ts * vec2(0, 2)).rgb;
    vec3 c = texture(uSrc, vUv + ts * vec2(2, 2)).rgb;
    vec3 d = texture(uSrc, vUv + ts * vec2(-2, 0)).rgb;
    vec3 e = texture(uSrc, vUv).rgb;
    vec3 f = texture(uSrc, vUv + ts * vec2(2, 0)).rgb;
    vec3 g = texture(uSrc, vUv + ts * vec2(-2, -2)).rgb;
    vec3 h = texture(uSrc, vUv + ts * vec2(0, -2)).rgb;
    vec3 i = texture(uSrc, vUv + ts * vec2(2, -2)).rgb;
    vec3 j = texture(uSrc, vUv + ts * vec2(-1, 1)).rgb;
    vec3 k = texture(uSrc, vUv + ts * vec2(1, 1)).rgb;
    vec3 l = texture(uSrc, vUv + ts * vec2(-1, -1)).rgb;
    vec3 m = texture(uSrc, vUv + ts * vec2(1, -1)).rgb;

    outColor = prefilter((j + k + l + m) * 0.125 +
                         (a + b + d + e) * 0.03125 +
                         (b + c + e + f) * 0.03125 +
                         (d + e + g + h) * 0.03125 +
                         (e + f + h + i) * 0.03125);
}
