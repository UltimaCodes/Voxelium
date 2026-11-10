#version 460 core

// bloom mix, white balance, exposure, filmic curve, vibrance, manual sRGB encode, manual so the
// output is identical on every driver instead of trusting GL_FRAMEBUFFER_SRGB support. Then grade
// (contrast + saturation), vignette, grain, and an optional unsharp pass to claw back the softness
// TAA leaves (done in LDR so bright HDR highlights ring).

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uHdr;
layout(binding = 2) uniform sampler2D uBloom;
layout(location = 0) uniform float uBloomStrength;
layout(location = 1) uniform float uExposure;
layout(location = 3) uniform float uSharpen; // 0 = off
layout(location = 4) uniform vec2 uTexel;    // 1/resolution
layout(location = 5) uniform float uVignette;    // 0 = off
layout(location = 6) uniform float uGrain;       // 0 = off
layout(location = 7) uniform float uTime;        // grain animation only
layout(location = 8) uniform float uSaturation;  // 1 = untouched
layout(location = 9) uniform float uContrast;    // 1 = untouched
layout(location = 10) uniform float uTemperature; // 0 = neutral, positive warm, negative cool
layout(location = 12) uniform float uScotopic;    // night rod-vision shift, 0 = off
layout(location = 14) uniform float uDither;      // 8-bit output dither, 0 = off

// Lottes' curve (GDC 2016): a toe for contrast and a long shoulder, so highlights roll off.
const float kToe = 1.24, kShoulder = 0.93, kHdrMax = 12.0, kMidIn = 0.18, kMidOut = 0.145;
const float kMaxD = pow(kHdrMax, kToe * kShoulder), kMidD = pow(kMidIn, kToe * kShoulder);
const float kMaxA = pow(kHdrMax, kToe), kMidA = pow(kMidIn, kToe);
const float kCurveB = (kMaxA * kMidOut - kMidA) / ((kMaxD - kMidD) * kMidOut);
const float kCurveC = (kMaxD * kMidA - kMaxA * kMidD * kMidOut) / ((kMaxD - kMidD) * kMidOut);

vec3 filmic(vec3 x) {
    vec3 z = pow(x, vec3(kToe));
    return min(z / (pow(z, vec3(kShoulder)) * kCurveB + kCurveC), vec3(1.0));
}

// Each channel through the curve alone turns bright greens yellow; half of the result keeps the
// colour's own hue. Then a lift for the low saturation midtones.
vec3 toneCurve(vec3 c) {
    c = max(c, vec3(0.0));
    float peak = max(max(c.r, c.g), max(c.b, 1e-5));
    vec3 keptHue = c * (filmic(vec3(peak)).r / peak);
    vec3 m = mix(filmic(c), keptHue, 0.5);
    float luma = dot(m, vec3(0.2126, 0.7152, 0.0722));
    float sat = max(max(m.r, m.g), m.b) - min(min(m.r, m.g), m.b);
    return max(mix(vec3(luma), m, 1.0 + 0.06 * (1.0 - sat)), vec3(0.0));
}

vec3 srgbEncode(vec3 c) {
    return mix(12.92 * c, 1.055 * pow(max(c, vec3(1e-6)), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// Ordered dither before the 8-bit write. A sky gradient spans a few hundred code values over
// a thousand pixels, so the steps land as visible bands, the one artefact in this renderer
// that is purely a quantisation problem. A triangular-PDF dither of one code value turns the
// step into noise below the threshold of vision, for about three instructions.
float triDither(vec2 p) {
    // two uniform hashes summed = triangular distribution, which is what makes the noise
    // symmetric about zero instead of biasing the image brighter
    float a = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    float b = fract(sin(dot(p + 17.0, vec2(39.3468, 11.135))) * 24634.6345);
    return a + b - 1.0;
}

// White balance belongs BEFORE the tonemap: it is a property of the light in the scene,
// not of the print. Channel gains rather than a Planckian-locus conversion, this is a
// slider nobody calibrates against a colour chart, and the gains keep luminance roughly
// put so the exposure slider does not have to be re-touched after moving it.
vec3 whiteBalance(vec3 c) {
    vec3 g = vec3(1.0 + 0.20 * uTemperature, 1.0 + 0.02 * uTemperature,
                  1.0 - 0.20 * uTemperature);
    return c * g;
}

// Scotopic (rod) vision: below about 0.01 cd/m2 the cones stop contributing, so night
// vision is desaturated and shifted blue, the Purkinje effect. Worth having because it
// makes a frame READ as night at a higher actual brightness than a purely dark frame does,
// which is the opposite of the trap this engine fell into (crank the fill until you can see,
// then wonder why night looks like overcast afternoon).
//
// Applied in LDR after the tonemap and keyed off the pixel's own luminance, so a lamp or a
// moonlit rim keeps its colour and only the genuinely dark parts of the frame desaturate.
vec3 scotopic(vec3 c, float amount) {
    if (amount <= 0.0)
        return c;
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    // full effect in the deep shadows, none by the time anything is even dimly lit. Twice the
    // slider, so the default 0.25 already turns a moonlit field olive-grey, not green.
    float rod = (1.0 - smoothstep(0.0, 0.30, luma)) * min(amount * 2.0, 1.0);
    vec3 night = vec3(luma) * vec3(0.88, 0.96, 1.14); // desaturated, a little cool
    return mix(c, night, rod);
}

vec3 tonemap(vec3 hdr) {
    return srgbEncode(toneCurve(whiteBalance(hdr) * uExposure));
}

// Contrast pivots on 0.5 because it runs after the sRGB encode, where mid grey sits there
// rather than at 0.18. Saturation is against Rec.709 luma so a desaturating pull does not
// change how bright anything looks.
vec3 grade(vec3 c) {
    c = (c - 0.5) * uContrast + 0.5;
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luma), c, uSaturation);
}

float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec3 hdr = texture(uHdr, vUv).rgb + texture(uBloom, vUv).rgb * uBloomStrength;
    vec3 c = tonemap(hdr);

    if (uSharpen > 0.0) {
        vec3 l = tonemap(texture(uHdr, vUv - vec2(uTexel.x, 0.0)).rgb);
        vec3 r = tonemap(texture(uHdr, vUv + vec2(uTexel.x, 0.0)).rgb);
        vec3 u = tonemap(texture(uHdr, vUv - vec2(0.0, uTexel.y)).rgb);
        vec3 d = tonemap(texture(uHdr, vUv + vec2(0.0, uTexel.y)).rgb);

        // Contrast-adaptive, not a plain unsharp mask. The old version added the laplacian
        // straight on, which overshoots at every edge, and an overshoot reads as a bright
        // halo along terrain silhouettes, which at render scale below 1 is a worse artefact
        // than the softness it was meant to fix.
        //
        // Clamping the result into the 4-neighbour range makes overshoot impossible: a pixel
        // can be pushed toward the extremes of its own neighbourhood but never past them. So
        // edges gain local contrast, flat areas gain nothing, and nothing rings. The centre
        // is folded into the bounds so a pixel that is ALREADY an extreme (a lamp, a star,
        // a specular hit) is not dragged back toward its darker surroundings.
        vec3 mn = min(min(l, r), min(u, d));
        vec3 mx = max(max(l, r), max(u, d));
        vec3 sharp = c + (c * 4.0 - l - r - u - d) * uSharpen;
        c = clamp(sharp, min(mn, c), max(mx, c));
    }

    // Before the grade, so the saturation slider still has the last word on the result.
    c = scotopic(c, uScotopic);
    c = grade(c);

    if (uVignette > 0.0) {
        // Aspect-corrected, so the falloff is an ellipse that matches the frame instead of
        // a circle that clips the short edge first. Squared smoothstep keeps the centre
        // completely untouched, a vignette you can see starting is a vignette set wrong.
        vec2 d = (vUv - 0.5) * vec2(uTexel.y / max(uTexel.x, 1e-6), 1.0);
        float r = length(d) * 1.41421356;
        float v = smoothstep(0.55, 1.15, r);
        c *= 1.0 - uVignette * v * v;
    }

    if (uGrain > 0.0) {
        // Scaled by (1 - luma): film grain lives in the midtones and shadows, and grain in
        // a blown highlight reads as sensor noise instead. Animated, so this is the one
        // effect that makes consecutive frames differ, the offscreen harnesses leave it
        // at 0 for exactly the reason TAA is off there.
        float n = hash21(vUv * (1.0 / max(uTexel.x, 1e-6)) + fract(uTime) * 137.0) - 0.5;
        float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c += n * uGrain * 0.12 * (1.0 - luma);
    }

    // Last thing before the 8-bit write, so it dithers the value that is actually quantised.
    if (uDither > 0.0)
        c += triDither(gl_FragCoord.xy) * (uDither / 255.0);
    outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
