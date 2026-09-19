#version 460 core

// Depth only, except that cutout blocks have to be alpha-tested here as well. Leaves are
// full of holes; if the shadow pass ignores that, a canopy still lays down a solid slab of
// shadow and none of the light coming through the gaps ever shows on the ground. The
// dappled shade under a tree IS this discard.

in vec2 vUv;
in vec2 vGap;
flat in uint vLayer;

layout(binding = 0) uniform sampler2DArray uBlockTex;
// Share of a canopy, by area, that lets sunlight straight through. 0 = only the texture's own
// holes.
layout(location = 7) uniform float uLeafGap;

// leaves; keep in step with texLayer in texlayer.glsl
const uint kLeavesLayer = 7u;

float gapHash(vec2 p) {
    p = fract(p * vec2(0.1031, 0.1030));
    p += dot(p, p.yx + 33.33);
    return fract((p.x + p.y) * p.x);
}

float gapNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(gapHash(i), gapHash(i + vec2(1, 0)), f.x),
               mix(gapHash(i + vec2(0, 1)), gapHash(i + vec2(1, 1)), f.x), f.y);
}

void main() {
    if (vLayer != kLeavesLayer)
        return;
    if (texture(uBlockTex, vec3(vUv, float(vLayer))).a < 0.5)
        discard;
    // The tile's own holes are hashed per block, so the odds of one lining up through a thick
    // canopy fall off geometrically and the ground under a tree is a solid slab of shade.
    // Gaps keyed to the light plane fix that: they depend only on where a ray crosses the
    // plane perpendicular to the light, so every leaf layer along it agrees and a gap is a
    // real column of light all the way to the ground (and a shaft for the volumetrics).
    // Cells ~3 blocks across, two octaves so the edges are ragged rather than blobby.
    vec2 g = vGap / 3.0;
    float n = gapNoise(g) * 0.65 + gapNoise(g * 2.7 + 11.0) * 0.35;
    if (n > 1.0 - uLeafGap * 1.6)
        discard;
}
