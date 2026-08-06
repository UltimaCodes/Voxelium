#version 460 core

layout(location = 6) uniform sampler2DArray uBlockTex;

in vec2 vUv;
in vec4 vTint;
flat in float vLayer;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 rgb = vTint.rgb;
    if (vLayer >= 0.0) {
        // Sample a small patch of the block's own texture rather than its average colour,
        // so a shard off a brick reads as brick and one off gravel reads as gravel.
        rgb *= texture(uBlockTex, vec3(vUv * 0.35 + 0.3, vLayer)).rgb;
    } else {
        // droplet: round it off so a splash is not a cloud of squares
        if (dot(vUv - 0.5, vUv - 0.5) > 0.25)
            discard;
    }
    outColor = vec4(rgb, vTint.a);
}
