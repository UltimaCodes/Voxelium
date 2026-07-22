#version 460 core

in vec3 vWorldRel;
flat in vec3 vAlbedo;
flat in uint vDir;
flat in vec2 vLight;

layout(location = 0) out vec4 outColor; // HDR

layout(location = 4) uniform vec3 uSunL;      // direction toward the sun
layout(location = 5) uniform float uShadowsOn;
layout(location = 6) uniform mat4 uCascadeVp[3]; // consumes 6..17
layout(location = 18) uniform vec3 uCascadeSplit; // radial split distances
layout(location = 19) uniform vec3 uCascadeTexel; // world units per shadow texel

layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;

const vec3 kNormal[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));
const float kShade[6] = float[6](0.80, 0.80, 1.00, 0.50, 0.65, 0.65);

// fixed 16-tap Poisson disk — a soft penumbra without per-pixel rotation, so shadow
// edges don't crawl/shimmer when the camera moves (rotated PCF looks softer but needs
// TAA to be stable; fixed taps stay put).
const vec2 kPoisson[16] = vec2[16](
    vec2(-0.942, -0.399), vec2(0.946, -0.769), vec2(-0.094, -0.929), vec2(0.345, 0.294),
    vec2(-0.916, 0.458), vec2(-0.815, -0.879), vec2(-0.383, 0.277), vec2(0.975, 0.756),
    vec2(0.443, -0.975), vec2(0.537, -0.474), vec2(-0.265, -0.419), vec2(0.792, 0.191),
    vec2(-0.242, 0.997), vec2(-0.814, 0.914), vec2(0.200, 0.786), vec2(0.144, -0.141));

float sampleCascade(vec3 pos, vec3 N, int c) {
    // normal-offset bias scaled by the cascade texel kills acne without peter panning
    vec4 sc = uCascadeVp[c] * vec4(pos + N * uCascadeTexel[c] * 1.5, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float depth = sc.z - 0.0016;
    float radius = 2.2 / 1536.0;
    float sum = 0.0;
    for (int i = 0; i < 16; ++i)
        sum += texture(uShadowMap, vec4(uv + kPoisson[i] * radius, float(c), depth));
    return sum / 16.0;
}

float sunShadow(vec3 pos, vec3 N) {
    if (uShadowsOn < 0.5)
        return 1.0;
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    float sh = sampleCascade(pos, N, c);
    // cross-fade into the next cascade near the split so there's no lighting seam
    float splitEnd = c == 0 ? uCascadeSplit.x : (c == 1 ? uCascadeSplit.y : uCascadeSplit.z);
    float blendStart = splitEnd * 0.82;
    if (c < 2 && d > blendStart)
        sh = mix(sh, sampleCascade(pos, N, c + 1), (d - blendStart) / (splitEnd - blendStart));
    // and fade shadows out at the very edge of the last cascade instead of a hard cut
    sh = mix(sh, 1.0, smoothstep(uCascadeSplit.z * 0.9, uCascadeSplit.z, d));
    return sh;
}

void main() {
    vec3 N = kNormal[vDir];
    float blockL = vLight.x;
    float skyL = vLight.y;

    float ndl = max(dot(N, uSunL), 0.0);
    float shadow = ndl > 0.0 ? sunShadow(vWorldRel, N) : 0.0;
    float skyAccess = skyL * skyL; // voxel sky light gates how much sun reaches here

    // warm strong sun vs cool dim sky ambient — the contrast is what keeps the scene
    // from going milky once tonemapped. the small floor stops fully-enclosed faces
    // (interior leaf pockets, deep cave walls) from crushing to pure black.
    vec3 direct = vec3(2.05, 1.62, 1.08) * (ndl * shadow * skyAccess);
    vec3 ambient = vec3(0.26, 0.36, 0.55) * (0.10 + 0.90 * skyL) + vec3(0.035, 0.04, 0.05);
    vec3 torch = vec3(1.35, 0.95, 0.55) * (blockL * blockL * 1.6);

    outColor = vec4(vAlbedo * kShade[vDir] * (direct + ambient + torch), 1.0);
}
