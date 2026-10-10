#version 460 core

// Multiplies ambient occlusion onto the OPAQUE scene, BEFORE water is drawn on top of it.
//
// This used to happen in composite.frag, at the very end of the frame, applied to the whole
// finished image in one multiply. That is wrong for water specifically: water never writes
// depth (see FrameRenderer.cpp's drawWater call, depth test but no write, so refraction can
// still read the pre-water depth), so SSAO is computed from the LAKE BED's geometry sitting
// underneath it, not from water's own surface. Applying that AO to the whole frame at the
// end meant water pixels got darkened by whatever occlusion the bed underneath them had,
// tracing the bed's own terrain contours as if they were shadows ON the water surface.
// Owner report: dense, dark, wavy contour tracery covering entire lake surfaces at every
// distance and viewing angle. Confirmed by isolation, every term in water.frag individually
// neutralized (wave normal, reflection, refraction, shadow, even the whole output colour
// forced flat) with the pattern completely unchanged; disabling SSAO outright made it vanish
// instantly. The fix is structural: apply AO to the opaque scene before water exists in the
// colour buffer at all, so there is nothing for water to inherit.
//
// Rendered with GL_DST_COLOR/GL_ZERO blending onto the HDR buffer, so this shader only needs
// to output the AO factor itself (broadcast to all three channels), the blend hardware does
// the actual multiply against whatever is already there, which is what lets this read the
// opaque snapshot for the emitter check without also having to read-and-write the same
// texture it is being composited into.

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uScene; // opaque snapshot, for the emitter check only
layout(binding = 1) uniform sampler2D uAo;

void main() {
    vec3 color = texture(uScene, vUv).rgb;
    // Same exception the old composite.frag step had: a surface producing its own light is
    // not RECEIVING ambient light, so occluding it is meaningless, unguarded, this put a
    // dark band along the bottom edge of every lamp. Same threshold, tuned the same way
    // (1.0 leaked onto 2.4% of a daylight frame, this leaves 0).
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float emitting = smoothstep(2.0, 3.2, lum);
    float ao = mix(texture(uAo, vUv).r, 1.0, emitting);
    outColor = vec4(vec3(ao), 1.0);
}
