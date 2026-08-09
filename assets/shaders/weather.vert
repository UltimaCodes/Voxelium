#version 460 core

// Precipitation, generated entirely here: no CPU simulation, no buffer, no upload.
//
// The drops live in a fixed-size box that FOLLOWS the camera, and each one's position is a
// pure function of its instance id and the clock. That is what makes it free — a particle
// system would have to simulate and re-upload tens of thousands of drops every frame, and
// the existing pool is 4096 with real per-drop physics it does not need here.
//
// The box wraps: a drop falls, and its height is taken modulo the box height, so it
// reappears at the top. Because the box is anchored to the camera's cell rather than to the
// camera itself, the drops do not slide with the player — walking moves you THROUGH the
// field instead of dragging it along, which is the difference between rain and a screensaver.

layout(location = 0) uniform mat4 uViewProj;
layout(location = 4) uniform vec3 uRight; // camera basis for the billboard
layout(location = 5) uniform vec3 uUp;
layout(location = 6) uniform vec3 uCamFrac; // camera position within its own cell
layout(location = 7) uniform float uTime;
layout(location = 8) uniform float uSnow;   // 0 rain .. 1 snow
layout(location = 9) uniform float uExtent; // half-size of the box, world units

out float vFade;
out vec2 vUv;

const vec2 kQuad[6] =
    vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

// cheap 3-out hash; the drops only need to look unrelated to each other
vec3 hash31(float n) {
    vec3 p = fract(vec3(n * 0.1031, n * 0.1030, n * 0.0973));
    p += dot(p, p.yzx + 33.33);
    return fract((p.xxy + p.yzz) * p.zyx);
}

void main() {
    const int drop = gl_VertexID / 6;
    const int corner = gl_VertexID % 6;
    vec3 h = hash31(float(drop) + 0.5);

    // Rain falls fast and straight; snow drifts and tumbles, which is most of what separates
    // the two visually — the shapes matter far less than the motion.
    float speed = mix(26.0, 2.4, uSnow) * (0.75 + 0.5 * h.z);
    float boxH = uExtent * 2.0;
    float y = uExtent - mod(h.y * boxH + uTime * speed, boxH);

    vec3 pos = vec3((h.x * 2.0 - 1.0) * uExtent, y, (h.z * 2.0 - 1.0) * uExtent);
    // Drift. Snow wanders on two out-of-phase sines so it never reads as a repeating pattern;
    // rain gets a slight slant so it does not look like it is falling in a vacuum.
    float drift = mix(0.25, 2.2, uSnow);
    pos.x += sin(uTime * (0.6 + h.x) + h.y * 6.28) * drift;
    pos.z += cos(uTime * (0.5 + h.y) + h.x * 6.28) * drift;
    // Anchor to the camera's CELL, not the camera: subtracting the fractional part is what
    // stops the whole field sliding along with the player.
    pos -= uCamFrac;

    // A rain drop is a vertical streak, a snowflake is a small square.
    vec2 c = kQuad[corner];
    float wide = mix(0.016, 0.030, uSnow);
    float tall = mix(0.55, 0.030, uSnow);
    vec3 world = pos + uRight * (c.x * wide) + uUp * (c.y * tall);

    vUv = c * 0.5 + 0.5;
    // Fade at the edges of the box so drops appear and vanish gradually instead of popping
    // in at a hard boundary, which is the giveaway that the field is a box at all.
    float r = length(pos.xz) / uExtent;
    vFade = 1.0 - smoothstep(0.55, 1.0, r);
    // And fade the ones right on top of the lens. A billboard of fixed world size covers
    // more screen the closer it gets, so without this the handful of drops nearest the
    // camera dominate the frame as big soft blobs — which is what snow looked like before.
    // A real drop that close is far out of focus anyway.
    vFade *= smoothstep(0.6, 2.6, length(pos));
    gl_Position = uViewProj * vec4(world, 1.0);
}
