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
out float vTone; // per-drop brightness, so a sheet of rain is not one streak repeated

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
    vec3 h2 = hash31(float(drop) * 1.7 + 41.0); // independent of h: speed, length, brightness

    // Rain falls fast and straight; snow drifts and tumbles, which is most of what separates
    // the two visually - the shapes matter far less than the motion.
    float speed = mix(26.0, 2.4, uSnow) * (0.75 + 0.5 * h2.x);
    float boxH = uExtent * 2.0;
    float y = uExtent - mod(h.y * boxH + uTime * speed, boxH);

    vec3 pos = vec3((h.x * 2.0 - 1.0) * uExtent, y, (h.z * 2.0 - 1.0) * uExtent);
    // Snow wanders on two out-of-phase sines so it never reads as a repeating pattern.
    float drift = mix(0.0, 2.2, uSnow);
    pos.x += sin(uTime * (0.6 + h.x) + h.y * 6.28) * drift;
    pos.z += cos(uTime * (0.5 + h.y) + h.x * 6.28) * drift;

    // Rain falls along a slant, and that same slant is the direction its streak points, so a
    // drop travels down a line instead of hanging as a vertical dash. The lean eases with a
    // slow gust. The horizontal offset is a function of height, so the trajectory is straight;
    // it is centred on the middle of the box so the field does not shear off to one side.
    float lean = 0.10 + 0.07 * sin(uTime * 0.31 + 1.3);
    vec3 fall = normalize(vec3(lean, -1.0, lean * 0.4));
    pos.xz += vec2(lean, lean * 0.4) * (-y);

    // Anchor to the camera's CELL, not the camera: subtracting the fractional part is what
    // stops the whole field sliding along with the player.
    pos -= uCamFrac;

    vec2 c = kQuad[corner];
    vec3 world;
    if (uSnow < 0.5) {
        // A ribbon along the fall direction, turned to face the viewer about that axis only.
        // The old streak used the camera's own up vector, so looking up or down (or leaning
        // the view) tilted every drop with the camera instead of keeping it pointing down.
        vec3 axis = -fall;
        vec3 side = cross(axis, pos);
        float sl = length(side);
        // looking straight along a streak it is end-on: fall back to any perpendicular
        side = sl > 1e-3 ? side / sl : uRight;
        float dist = length(pos);
        float len = mix(0.45, 1.25, h2.y) * (speed / 26.0);
        // at least about a pixel wide however far away, or distant rain vanishes
        float wide = max(0.010 + 0.012 * h2.z, dist * 0.0011);
        float t = c.y * 0.5 + 0.5; // 0 head, 1 tail
        world = pos + axis * (t * len) + side * (c.x * wide);
        vUv = vec2(c.x * 0.5 + 0.5, t);
    } else {
        // a snowflake is a small camera-facing square, its size varying per flake
        float size = 0.022 + 0.022 * h2.y;
        world = pos + uRight * (c.x * size) + uUp * (c.y * size);
        vUv = c * 0.5 + 0.5;
    }
    vTone = 0.35 + 0.65 * h2.z;

    // Fade at the edges of the box so drops appear and vanish gradually instead of popping
    // in at a hard boundary, which is the giveaway that the field is a box at all.
    float r = length(pos.xz) / uExtent;
    vFade = 1.0 - smoothstep(0.55, 1.0, r);
    // And fade the ones right on top of the lens. A billboard of fixed world size covers
    // more screen the closer it gets, so without this the handful of drops nearest the
    // camera dominate the frame as big soft blobs.
    vFade *= smoothstep(0.6, 2.6, length(pos));
    gl_Position = uViewProj * vec4(world, 1.0);
}
