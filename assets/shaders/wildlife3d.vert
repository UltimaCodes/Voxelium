#version 460 core

// Creatures with actual volume. A camera-facing sprite is a flat cut-out: from the side or from
// above it turns edge-on and there is nothing left to see, which is what the first version of
// this looked like. Each creature here is a handful of small boxes (a body, a head, wings that
// beat, a tail that wags) generated from the same vertex-pulled record the glows use, so they
// have a front and a back and read as animals from any angle.
//
// One creature is kParts boxes of 36 vertices each. Boxes a kind does not use collapse to a
// degenerate triangle. Everything is built in the creature's own frame, x forward, y up,
// z to its right, and turned into the world by its heading.
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
const int kButterfly = 1, kGull = 3, kFish = 4, kLeaf = 5, kBird = 8, kCrab = 9, kTurtle = 10, kJelly = 11;
const int kSheep = 12, kCow = 13, kPig = 14, kChicken = 15, kRabbit = 16, kDeer = 17, kFox = 18, kFrog = 19, kBee = 20, kBat = 21;

// set by a model that glows (a jellyfish): its colour is then used flat, not shaded
float gEmit = 0.0;

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

// Long axis along z, moving along x: a crab walks sideways. Legs bob in alternating pairs and the
// claws open and close.
Part crab(int p, float ph) {
    if (p == 0) return Part(vec3(0.0), vec3(0.30, 0.10, 0.44), mat3(1), vec3(1.0), true);
    if (p == 1 || p == 2) {
        float side = p == 1 ? 1.0 : -1.0;
        mat3 r = rotY(side * (0.10 + 0.30 * (0.5 + 0.5 * sin(ph * 0.6 + side))));
        return Part(vec3(0.38, 0.04, side * 0.34), vec3(0.22, 0.05, 0.07), r, vec3(1.15, 0.9, 0.85), true);
    }
    if (p >= 3 && p <= 6) {
        int k = p - 3;
        float side = (k & 1) == 0 ? 1.0 : -1.0;
        float xo = k < 2 ? 0.12 : -0.14;
        float bob = 0.035 * sin(ph * 2.0 + float(k) * 1.9);
        return Part(vec3(xo, -0.07 + bob, side * 0.55), vec3(0.035, 0.03, 0.20), rotX(side * 0.35), vec3(0.85), true);
    }
    if (p == 7) return Part(vec3(0.27, 0.17, 0.0), vec3(0.03, 0.05, 0.22), mat3(1), vec3(0.06), true);
    return none();
}

Part turtle(int p, float ph) {
    float fl = sin(ph) * 0.6;
    if (p == 0) return Part(vec3(0.0, 0.06, 0.0), vec3(0.50, 0.13, 0.40), mat3(1), vec3(0.85, 0.95, 0.7), true);
    if (p == 1) return Part(vec3(0.0, -0.06, 0.0), vec3(0.44, 0.05, 0.34), mat3(1), vec3(1.6, 1.5, 1.05), true);
    if (p == 2) return Part(vec3(0.62, 0.02, 0.0), vec3(0.14, 0.09, 0.09), mat3(1), vec3(1.25, 1.2, 0.95), true);
    if (p == 3 || p == 4) {
        float side = p == 3 ? 1.0 : -1.0;
        mat3 r = rotX(side * fl);
        return Part(r * vec3(0.30, -0.02, side * 0.46), vec3(0.16, 0.03, 0.26), r, vec3(1.1, 1.1, 0.85), true);
    }
    if (p == 5 || p == 6) {
        float side = p == 5 ? 1.0 : -1.0;
        mat3 r = rotX(side * fl * 0.7);
        return Part(r * vec3(-0.38, -0.02, side * 0.30), vec3(0.10, 0.03, 0.14), r, vec3(1.05, 1.05, 0.85), true);
    }
    if (p == 7) return Part(vec3(-0.58, 0.0, 0.0), vec3(0.08, 0.03, 0.04), mat3(1), vec3(1.0), true);
    return none();
}

// A bell that breathes, a bright core, and four tentacles that sway. Glows, so it is used flat.
Part jelly(int p, float ph) {
    gEmit = 1.0;
    float pulse = 0.5 + 0.5 * sin(ph * 2.0);
    if (p == 0) return Part(vec3(0.0, 0.22, 0.0), vec3(0.42 - 0.07 * pulse, 0.07, 0.42 - 0.07 * pulse), mat3(1), vec3(1.0), true);
    if (p == 1) return Part(vec3(0.0, 0.34, 0.0), vec3(0.32 - 0.06 * pulse, 0.07, 0.32 - 0.06 * pulse), mat3(1), vec3(1.05), true);
    if (p == 2) return Part(vec3(0.0, 0.44, 0.0), vec3(0.18, 0.05, 0.18), mat3(1), vec3(1.2), true);
    if (p == 3) return Part(vec3(0.0, 0.26, 0.0), vec3(0.16, 0.10, 0.16), mat3(1), vec3(1.9), true);
    if (p >= 4 && p <= 7) {
        int k = p - 4;
        float ang = float(k) * 1.5708 + 0.4;
        float sway = sin(ph * 1.3 + float(k) * 1.7) * 0.10;
        vec3 off = vec3(cos(ang) * 0.22 + sway, -0.28, sin(ang) * 0.22);
        return Part(off, vec3(0.02, 0.30, 0.02), rotZ(sway * 0.8), vec3(0.8), true);
    }
    return none();
}

// Sheep, cows, pigs, deer and foxes: a body, a head that drops to graze, something on the head
// (wool, horns, a snout, a neck), four legs in a diagonal gait, and a tail. Built standing on the
// creature's position. A negative phase means it is standing still, and how far through grazing.
Part quadruped(int p, float ph, int kind) {
    float bodyY = 0.62, legX = 0.27, legZ = 0.16, hipY = 0.38, legH = 0.18, legW = 0.06, grazeAng = 0.9;
    vec3 bodyH = vec3(0.42, 0.26, 0.28), headC = vec3(0.52, 0.78, 0.0), headH = vec3(0.14, 0.13, 0.11);
    vec3 headCol = vec3(0.32), legCol = vec3(0.32);
    vec3 tailC = vec3(-0.44, 0.66, 0.0), tailH = vec3(0.05, 0.07, 0.06), tailCol = vec3(1.0);
    float tailTilt = 0.0;
    vec3 xC = vec3(0.50, 0.88, 0.0), xH = vec3(0.11, 0.05, 0.12), xCol = vec3(1.0);
    if (kind == kCow) {
        bodyY = 0.82; bodyH = vec3(0.55, 0.30, 0.30); headC = vec3(0.66, 0.98, 0.0); headH = vec3(0.16, 0.16, 0.14);
        headCol = vec3(0.95); hipY = 0.52; legH = 0.26; legW = 0.08; legX = 0.38; legZ = 0.19; legCol = vec3(0.8);
        tailC = vec3(-0.57, 0.80, 0.0); tailH = vec3(0.03, 0.22, 0.03); tailCol = vec3(0.8); tailTilt = 0.15;
        xC = vec3(0.64, 1.15, 0.0); xH = vec3(0.03, 0.03, 0.21); xCol = vec3(1.9, 1.8, 1.5);
    } else if (kind == kPig) {
        bodyY = 0.46; bodyH = vec3(0.40, 0.21, 0.24); headC = vec3(0.48, 0.50, 0.0); headH = vec3(0.14, 0.13, 0.15);
        headCol = vec3(1.0); hipY = 0.26; legH = 0.13; legW = 0.07; legX = 0.25; legZ = 0.15; legCol = vec3(0.92);
        tailC = vec3(-0.42, 0.54, 0.0); tailH = vec3(0.03, 0.04, 0.03); tailTilt = 0.6; grazeAng = 0.6;
        xC = vec3(0.63, 0.46, 0.0); xH = vec3(0.03, 0.06, 0.07); xCol = vec3(0.85);
    } else if (kind == kDeer) {
        bodyY = 0.98; bodyH = vec3(0.44, 0.21, 0.17); headC = vec3(0.62, 1.42, 0.0); headH = vec3(0.13, 0.09, 0.08);
        headCol = vec3(1.0); hipY = 0.77; legH = 0.38; legW = 0.045; legX = 0.32; legZ = 0.11; legCol = vec3(0.82);
        tailC = vec3(-0.46, 1.08, 0.0); tailH = vec3(0.04, 0.06, 0.05); tailCol = vec3(1.5); grazeAng = 1.6;
        xC = vec3(0.50, 1.22, 0.0); xH = vec3(0.07, 0.17, 0.07); xCol = vec3(1.0);
    } else if (kind == kFox) {
        bodyY = 0.36; bodyH = vec3(0.30, 0.11, 0.11); headC = vec3(0.38, 0.44, 0.0); headH = vec3(0.11, 0.09, 0.10);
        headCol = vec3(1.0); hipY = 0.25; legH = 0.12; legW = 0.035; legX = 0.20; legZ = 0.08; legCol = vec3(0.22);
        tailC = vec3(-0.44, 0.40, 0.0); tailH = vec3(0.17, 0.07, 0.08); tailCol = vec3(1.05); tailTilt = 0.25; grazeAng = 0.5;
        xC = vec3(0.51, 0.41, 0.0); xH = vec3(0.06, 0.04, 0.045); xCol = vec3(1.15);
    }
    bool walking = ph >= 0.0;
    float swing = walking ? sin(ph) * 0.55 : 0.0;
    float graze = walking ? 0.0 : smoothstep(0.15, 0.85, 0.5 + 0.5 * sin((-ph - 1.0) * 2.0));
    vec3 pivot = vec3(bodyH.x * 0.85, bodyY + bodyH.y * 0.6, 0.0);
    mat3 hr = rotZ(-graze * grazeAng);
    if (p == 0) return Part(vec3(0.0, bodyY, 0.0), bodyH, mat3(1), vec3(1.0), true);
    if (p == 1) return Part(pivot + hr * (headC - pivot), headH, hr, headCol, true);
    if (p == 2) return Part(pivot + hr * (xC - pivot), xH, hr, xCol, true);
    if (p >= 3 && p <= 6) {
        int k = p - 3;
        float sx = k < 2 ? 1.0 : -1.0;
        float sz = (k & 1) == 0 ? 1.0 : -1.0;
        float s = (k == 0 || k == 3) ? 1.0 : -1.0;
        mat3 r = rotZ(swing * s);
        vec3 hip = vec3(sx * legX, hipY, sz * legZ);
        return Part(hip + r * vec3(0.0, -legH, 0.0), vec3(legW, legH, legW), r, legCol, true);
    }
    if (p == 7) {
        mat3 r = rotZ(tailTilt) * rotY(sin(abs(ph) * 0.7) * 0.3);
        return Part(tailC, tailH, r, tailCol, true);
    }
    return none();
}

Part rabbit(int p, float ph) {
    if (p == 0) return Part(vec3(0.0, 0.20, 0.0), vec3(0.16, 0.12, 0.11), mat3(1), vec3(1.0), true);
    if (p == 1) return Part(vec3(0.17, 0.30, 0.0), vec3(0.08, 0.08, 0.07), mat3(1), vec3(1.0), true);
    if (p == 2 || p == 3) {
        float side = p == 2 ? 1.0 : -1.0;
        float twitch = ph < 0.0 ? sin(-ph * 3.0 + side) * 0.15 : 0.0;
        return Part(vec3(0.13, 0.44, side * 0.035), vec3(0.02, 0.09, 0.025), rotZ(0.25 + twitch), vec3(1.0), true);
    }
    if (p == 4) return Part(vec3(0.10, 0.06, 0.0), vec3(0.03, 0.06, 0.07), mat3(1), vec3(0.9), true);
    if (p == 5 || p == 6) {
        float side = p == 5 ? 1.0 : -1.0;
        return Part(vec3(-0.08, 0.06, side * 0.075), vec3(0.08, 0.05, 0.03), mat3(1), vec3(0.95), true);
    }
    if (p == 7) return Part(vec3(-0.17, 0.24, 0.0), vec3(0.035, 0.035, 0.035), mat3(1), vec3(1.5), true);
    return none();
}

Part chicken(int p, float ph) {
    bool walking = ph >= 0.0;
    float peck = walking ? 0.0 : smoothstep(0.6, 0.95, 0.5 + 0.5 * sin((-ph - 1.0) * 6.0));
    vec3 pivot = vec3(0.10, 0.40, 0.0);
    mat3 hr = rotZ(-peck * 1.1);
    float swing = walking ? sin(ph) * 0.6 : 0.0;
    if (p == 0) return Part(vec3(0.0, 0.30, 0.0), vec3(0.16, 0.13, 0.11), mat3(1), vec3(1.0), true);
    if (p == 1) return Part(pivot + hr * (vec3(0.14, 0.48, 0.0) - pivot), vec3(0.07, 0.07, 0.06), hr, vec3(1.0), true);
    if (p == 2) return Part(pivot + hr * (vec3(0.22, 0.47, 0.0) - pivot), vec3(0.035, 0.02, 0.02), hr, vec3(1.5, 1.0, 0.2), true);
    if (p == 3) return Part(pivot + hr * (vec3(0.14, 0.57, 0.0) - pivot), vec3(0.045, 0.03, 0.012), hr, vec3(1.5, 0.25, 0.2), true);
    if (p == 4 || p == 5) {
        float side = p == 4 ? 1.0 : -1.0;
        mat3 r = rotZ(swing * side);
        return Part(vec3(0.0, 0.17, side * 0.05) + r * vec3(0.0, -0.085, 0.0), vec3(0.015, 0.085, 0.015), r, vec3(1.4, 1.0, 0.3), true);
    }
    if (p == 6) return Part(vec3(-0.01, 0.31, 0.0), vec3(0.13, 0.08, 0.125), mat3(1), vec3(0.92), true);
    if (p == 7) return Part(vec3(-0.17, 0.40, 0.0), vec3(0.05, 0.08, 0.07), rotZ(0.4), vec3(0.95), true);
    return none();
}

Part frog(int p, float ph) {
    if (p == 0) return Part(vec3(0.0, 0.09, 0.0), vec3(0.12, 0.06, 0.10), mat3(1), vec3(1.0), true);
    if (p == 1) return Part(vec3(0.11, 0.12, 0.0), vec3(0.06, 0.05, 0.09), mat3(1), vec3(1.05), true);
    if (p == 2 || p == 3) {
        float side = p == 2 ? 1.0 : -1.0;
        return Part(vec3(0.12, 0.18, side * 0.055), vec3(0.025), mat3(1), vec3(0.35), true);
    }
    if (p == 4 || p == 5) {
        float side = p == 4 ? 1.0 : -1.0;
        return Part(vec3(-0.07, 0.05, side * 0.11), vec3(0.08, 0.03, 0.04), mat3(1), vec3(0.9), true);
    }
    if (p == 6 || p == 7) {
        float side = p == 6 ? 1.0 : -1.0;
        return Part(vec3(0.08, 0.035, side * 0.08), vec3(0.02, 0.035, 0.02), mat3(1), vec3(0.9), true);
    }
    return none();
}

Part bee(int p, float ph) {
    if (p == 0) return Part(vec3(0.0), vec3(0.32, 0.20, 0.20), mat3(1), vec3(1.0), true);
    if (p == 1) return Part(vec3(-0.08, 0.0, 0.0), vec3(0.05, 0.205, 0.205), mat3(1), vec3(0.08), true);
    if (p == 2) return Part(vec3(0.12, 0.0, 0.0), vec3(0.05, 0.205, 0.205), mat3(1), vec3(0.08), true);
    if (p == 3) return Part(vec3(0.38, 0.0, 0.0), vec3(0.10, 0.13, 0.13), mat3(1), vec3(0.12), true);
    if (p == 4 || p == 5) {
        float side = p == 4 ? 1.0 : -1.0;
        mat3 r = rotX(side * mix(0.2, 1.0, 0.5 + 0.5 * sin(ph)));
        return Part(r * vec3(0.02, 0.0, side * 0.24), vec3(0.16, 0.01, 0.22), r, vec3(0.85, 1.1, 5.0), true);
    }
    if (p == 6) return Part(vec3(-0.36, 0.0, 0.0), vec3(0.05), mat3(1), vec3(0.08), true);
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
    else if (kind == kBird || kind == kBat) pt = flyer(p, ph, false);
    else if (kind == kFish) pt = fish(p, ph);
    else if (kind == kCrab) pt = crab(p, ph);
    else if (kind == kTurtle) pt = turtle(p, ph);
    else if (kind == kJelly) pt = jelly(p, ph);
    else if (kind == kSheep || kind == kCow || kind == kPig || kind == kDeer || kind == kFox) pt = quadruped(p, ph, kind);
    else if (kind == kRabbit) pt = rabbit(p, ph);
    else if (kind == kChicken) pt = chicken(p, ph);
    else if (kind == kFrog) pt = frog(p, ph);
    else if (kind == kBee) pt = bee(p, ph);
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
    vec3 base = s.tint.rgb;
    if (kind == kFish && p <= 1) {
        // Countershading: a dark back and a pale, silvery belly, graded down the flanks (the
        // corners carry it, so the side faces interpolate it). One flat colour read as a tinted
        // block of water from above; this is what makes a real fish stand out from either side.
        // The belly is the tint desaturated and lifted, so it still dims with the time of day.
        float up = clamp(localPos.y / 0.22 * 0.5 + 0.5, 0.0, 1.0);
        vec3 belly = mix(base, vec3(dot(base, vec3(0.3333)) * 1.55), 0.65);
        base = mix(belly, base * 0.72, smoothstep(0.15, 0.85, up));
    }
    vColor = base * pt.col * mix(l, 1.0, gEmit);
    vAlpha = s.tint.a * (kind == kJelly ? 0.78 : 1.0);
    gl_Position = uViewProj * vec4(world, 1.0);
}
