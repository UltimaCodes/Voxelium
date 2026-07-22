#version 460 core

// depth-aware (bilateral) blur of the half-res AO: taps that straddle a depth edge are
// rejected, so AO no longer smears across silhouettes — that smearing was the "blur when
// you get close to a block". carries the view depth through in .g for the composite.

in vec2 vUv;
layout(location = 0) out vec2 outAoDepth;
layout(binding = 0) uniform sampler2D uSrc; // .r = ao, .g = view z

void main() {
    vec2 texel = 1.0 / vec2(textureSize(uSrc, 0));
    vec2 center = texture(uSrc, vUv).rg;
    float centerZ = center.g;
    float sum = 0.0;
    float wsum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            vec2 s = texture(uSrc, vUv + vec2(x, y) * texel).rg;
            // weight falls off fast with view-depth difference (0.5 units ~ one block)
            float w = exp(-abs(s.g - centerZ) * 2.0);
            sum += s.r * w;
            wsum += w;
        }
    outAoDepth = vec2(sum / max(wsum, 1e-4), centerZ);
}
