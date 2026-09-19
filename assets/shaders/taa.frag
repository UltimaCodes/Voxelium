#version 460 core

// Temporal anti-aliasing resolve. The projection is jittered a sub-pixel amount each frame
// (main.cpp), so accumulating frames over a static world converges to a supersampled,
// alias-free, shimmer-free image. Reprojection is exact because rendering is
// camera-relative: reconstruct the current pixel's world position from depth, shift it by
// how far the camera moved, and project it with last frame's view-proj to find where it
// was. A 3x3 neighbourhood colour clamp rejects the ghosting that reprojection alone
// leaves on disocclusions and moving edges.

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uCurrent; // this frame, post-composite HDR
layout(binding = 1) uniform sampler2D uHistory; // last frame's resolved output
layout(binding = 2) uniform sampler2D uDepth;

layout(location = 0) uniform vec2 uProjScale;   // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot;  // view -> camera-relative world
layout(location = 5) uniform mat4 uPrevViewProj; // last frame, unjittered
layout(location = 9) uniform vec3 uCamDelta;    // thisCamPos - prevCamPos
layout(location = 10) uniform float uHistoryValid;
layout(location = 11) uniform vec2 uTexel;      // (1/w, 1/h)
layout(location = 12) uniform float uCamY;      // camera world height, for the cloud layer

void main() {
    vec3 cur = texture(uCurrent, vUv).rgb;
    float rawD = texture(uDepth, vUv).r;
    vec2 ndc = vUv * 2.0 - 1.0;

    // current pixel's camera-relative world position (a far point for sky pixels, which
    // reproject by rotation only)
    vec3 worldRel;
    if (rawD <= 0.0) {
        // Sky is mostly cloud, and clouds are a real layer 420..560 up: moving vertically
        // shifts them on screen by parallax. Reprojecting as if at infinity (rotation only)
        // held last frame's clouds in place, so they looked glued to the camera while the
        // ground slid past. Put the pixel on the layer's middle instead.
        vec3 dir = normalize(uInvViewRot * vec3(ndc.x * uProjScale.x, ndc.y * uProjScale.y, -1.0));
        float dist = 10000.0;
        if (dir.y > 0.05) {
            float t = (490.0 - uCamY) / dir.y;
            if (t > 1.0)
                dist = min(t, 10000.0);
        }
        worldRel = dir * dist;
    } else {
        float viewZ = -uNear / max(rawD, 1e-7);
        worldRel = uInvViewRot * vec3(ndc.x * uProjScale.x * -viewZ,
                                      ndc.y * uProjScale.y * -viewZ, viewZ);
    }

    // where this point sat on screen last frame
    vec4 pc = uPrevViewProj * vec4(worldRel + uCamDelta, 1.0);
    vec2 prevUv = (pc.xy / pc.w) * 0.5 + 0.5;

    if (uHistoryValid < 0.5 || pc.w <= 0.0 || any(lessThan(prevUv, vec2(0.0))) ||
        any(greaterThan(prevUv, vec2(1.0)))) {
        outColor = vec4(cur, 1.0); // first frame / off-screen: nothing to blend
        return;
    }

    // 3x3 neighbourhood AABB of the current frame -> clamp history into it (anti-ghost)
    vec3 lo = cur, hi = cur;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            vec3 c = texture(uCurrent, vUv + vec2(x, y) * uTexel).rgb;
            lo = min(lo, c);
            hi = max(hi, c);
        }
    vec3 hist = clamp(texture(uHistory, prevUv).rgb, lo, hi);

    outColor = vec4(mix(cur, hist, 0.9), 1.0); // ~10 frames of accumulation
}
