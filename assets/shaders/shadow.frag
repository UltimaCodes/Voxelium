#version 460 core

// Depth only, except that cutout blocks have to be alpha-tested here as well. Leaves are
// full of holes; if the shadow pass ignores that, a canopy still lays down a solid slab of
// shadow and none of the light coming through the gaps ever shows on the ground. The
// dappled shade under a tree IS this discard.

in vec2 vUv;
flat in uint vLayer;

layout(binding = 0) uniform sampler2DArray uBlockTex;

// leaves; keep in step with texLayer in texlayer.glsl
const uint kLeavesLayer = 7u;

void main() {
    if (vLayer == kLeavesLayer && texture(uBlockTex, vec3(vUv, float(vLayer))).a < 0.5)
        discard;
}
