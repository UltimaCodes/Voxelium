#version 460 core

// Creatures with actual volume. A camera-facing sprite is a flat cut-out: from the side or from
// above it turns edge-on and there is nothing left to see, which is what the first version of
// this looked like. Each creature here is a handful of small boxes (a body, a head, wings that
// beat, a tail that wags) generated from the same vertex-pulled record the glows use, so they
// have a front and a back and read as animals from any angle.
//
// One creature is kParts boxes of 36 vertices each. Boxes a kind does not use collapse to a
// degenerate triangle. Everything is built in the creature's own frame - x forward, y up,
// z to its right - and turned into the world by its heading.
struct S {
    vec4 posSize; // camera-relative centre, then scale
    vec4 tint;    // rgb, then alpha (fades in and out by dithering)
    vec4 kindDir; // x = kind, yz = horizontal heading (world x, z), w = animation phase
};
layout(std430, binding = 0) readonly buffer Sprites { S sprites[]; };

layout(location = 0) uniform mat4 uViewProj;

out vec3 vColor;
out float vAlpha;

const int kParts = 8;
const int kVerts = kParts * 36;
const int kButterfly = 1, kGull = 3, kFish = 4, kLeaf = 5, kBird = 8;

// faces of a unit box: outward normal and the two tangents, u x v = n so the winding is CCW
const vec3 kN[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0),
                           vec3(0, 0, 1), vec3(0, 0, -1));
const vec3 kU[6] = vec3[6](vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 0, 1), vec3(1, 0, 0),
                           vec3(1, 0, 0), vec3(0, 1, 0));
const vec3 kV[6] = vec3[6](vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0), vec3(0, 0, 1),
                           vec3(0, 1, 0), vec3(1, 0, 0));
const vec2 kCorner[6] =
    vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

mat3 rotX(float a) { float c = cos(a), s = sin(a); return mat3(1, 0, 0, 0, c, s, 0, -s, c); }
mat3 rotY(float a) { float c = cos(a), s = sin(a); return mat3(c, 0, -s, 0, 1, 0, s, 0, c); }
mat3 rotZ(float a) { float c = cos(a), s = sin(a); return mat3(c, s, 0, -s, c, 0, 0, 0, 1); }

// One box of a creature: where its centre is, how big, how it is turned, and a colour factor.
struct Part {
    vec3 c;
    vec3 h;
    mat3 r;
    vec3 col;
    bool on;
};

Part none() { return Part(vec3(0), vec3(0), mat3(1), vec3(1), false); }

Part butterfly(int p, float ph) {
    // wings hinge on the body and beat up and down together
    float ang = mix(0.10, 1.15, 0.5 + 0.5 * sin(ph));
    if (p == 0) return Part(vec3(0), vec3(0.42, 0.07, 0.07), mat3(1), vec3(0.10), true);
    if (p == 1) return Part(vec3(0.52, 0.02, 0), vec3(0.10, 0.09, 0.09), mat3(1), vec3(0.10), true);
    if (p >= 2 && p <= 5) {
        float side = (p & 1) == 0 ? 1.0 : -1.0;
        bool front = p < 4;
        mat3 r = rotX(side * ang);
        vec3 hinge = vec3(front ? 0.10 : -0.20, 0.0, side * (front ? 0.36 : 0.28));
        vec3 hs = front ? vec3(0.30, 0.012, 0.36) : vec3(0.22, 0.012, 0.28);
        return Part(r * hinge, hs, r, front ? vec3(1.0) : vec3(0.78, 0.74, 0.90), true);
    }
    return none();
}

Part flyer(int p, float ph, bool gull) {
    float flap = sin(ph) * 0.55;
    vec3 dark = gull ? vec3(0.16) : vec3(1.0);
    if (p == 0) return Part(vec3(0), vec3(0.55, 0.13, 0.15), mat3(1), vec3(1.0), true);
    if (p == 1) return Part(vec3(0.66, 0.08, 0), vec3(0.13, 0.11, 0.11), mat3(1), vec3(1.0), true);
    if (p == 2) return Part(vec3(0.84, 0.05, 0), vec3(0.10, 0.03, 0.03), mat3(1),
                            gull ? vec3(1.4, 1.0, 0.25) : vec3(0.7), true);
    if (p == 3) return Part(vec3(-0.70, 0.0, 0), vec3(0.22, 0.03, 0.13), mat3(1), vec3(0.95), true);
    if (p >= 4) {
        float side = (p & 1) == 0 ? 1.0 : -1.0;
        bool outer = p >= 6;
        mat3 ri = rotX(side * flap);
        if (!outer)
            return Part(ri * vec3(0.05, 0.0, side * 0.42), vec3(0.22, 0.025, 0.42), ri, vec3(1.0), true);
        mat3 ro = rotX(side * flap * 1.7);
        return Part(ri * vec3(0.05, 0.0, side * 0.84) + ro * vec3(-0.05, 0.0, side * 0.36),
                    vec3(0.16, 0.02, 0.36), ro, dark, true);
    }
    return none();
}

Part fish(int p, float ph) {
    float wag = sin(ph) * 0.45;
    if (p == 0) return Part(vec3(-0.05, 0, 0), vec3(0.56, 0.22, 0.12), mat3(1), vec3(1.0), true);
    if (p == 1) return Part(vec3(0.58, -0.02, 0), vec3(0.16, 0.16, 0.10), mat3(1), vec3(0.95), true);
    if (p == 2) {
        mat3 r = rotY(wag);
        return Part(vec3(-0.62, 0, 0) + r * vec3(-0.14, 0, 0), vec3(0.16, 0.22, 0.025), r,
                    vec3(0.85, 0.80, 0.80), true);
    }
    if (p == 3) return Part(vec3(-0.05, 0.28, 0), vec3(0.22, 0.10, 0.02), mat3(1), vec3(0.85), true);
    if (p == 4 || p == 5) {
        float side = p == 4 ? 1.0 : -1.0;
        mat3 r = rotX(side * 0.5);
        return Part(vec3(0.22, -0.10, side * 0.15), vec3(0.10, 0.02, 0.12), r, vec3(0.85), true);
    }
    if (p == 6 || p == 7) {
        float side = p == 6 ? 1.0 : -1.0;
        return Part(vec3(0.60, 0.06, side * 0.105), vec3(0.04, 0.04, 0.02), mat3(1), vec3(0.03), true);
    }
    return none();
}

Part leaf(int p, float ph) {
    if (p != 0) return none();
    mat3 r = rotY(ph * 0.7) * rotX(sin(ph) * 1.1) * rotZ(ph * 0.5);
    return Part(vec3(0), vec3(0.5, 0.02, 0.32), r, vec3(1.0), true);
}

void main() {
    int creature = gl_VertexID / kVerts;
    int local = gl_VertexID % kVerts;
    int p = local / 36;
    int v = local % 36;
    S s = sprites[creature];
    int kind = int(s.kindDir.x + 0.5);
    float ph = s.kindDir.w;

    Part pt;
    if (kind == kButterfly) pt = butterfly(p, ph);
    else if (kind == kGull) pt = flyer(p, ph, true);
    else if (kind == kBird) pt = flyer(p, ph, false);
    else if (kind == kFish) pt = fish(p, ph);
    else pt = leaf(p, ph);

    if (!pt.on) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // clipped: a degenerate triangle
        vColor = vec3(0.0);
        vAlpha = 0.0;
        return;
    }

    int face = v / 6;
    vec2 cc = kCorner[v % 6];
    vec3 localPos = pt.c + pt.r * ((kN[face] + kU[face] * cc.x + kV[face] * cc.y) * pt.h);
    vec3 localN = pt.r * kN[face];

    vec2 hd = s.kindDir.yz;
    hd = dot(hd, hd) > 1e-6 ? normalize(hd) : vec2(1.0, 0.0);
    vec3 fwd = vec3(hd.x, 0.0, hd.y);
    vec3 side = vec3(-hd.y, 0.0, hd.x);
    vec3 world = s.posSize.xyz + (fwd * localPos.x + vec3(0, 1, 0) * localPos.y + side * localPos.z) * s.posSize.w;
    vec3 wn = fwd * localN.x + vec3(0, 1, 0) * localN.y + side * localN.z;

    // a fixed soft key from above and to one side; the tint already carries the time of day
    float l = 0.52 + 0.36 * wn.y + 0.14 * dot(wn, normalize(vec3(0.4, 0.0, 0.3)));
    vColor = s.tint.rgb * pt.col * l;
    vAlpha = s.tint.a;
    gl_Position = uViewProj * vec4(world, 1.0);
}
