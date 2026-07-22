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

float sunShadow(vec3 pos, vec3 N) {
    if (uShadowsOn < 0.5)
        return 1.0;
    float d = length(pos);
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    if (d >= uCascadeSplit.z)
        return 1.0;
    // normal-offset bias scaled by the cascade's texel size kills acne without peter
    // panning on axis-aligned voxel faces
    vec4 sc = uCascadeVp[c] * vec4(pos + N * uCascadeTexel[c] * 1.5, 1.0);
    vec3 uvz = vec3(sc.xy * 0.5 + 0.5, sc.z); // glm ortho is already 0..1 in z
    float texel = 1.0 / 1536.0;
    float sum = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            sum += texture(uShadowMap,
                           vec4(uvz.xy + vec2(dx, dy) * texel, float(c), uvz.z - 0.0012));
    return sum / 9.0;
}

void main() {
    vec3 N = kNormal[vDir];
    float blockL = vLight.x;
    float skyL = vLight.y;

    float ndl = max(dot(N, uSunL), 0.0);
    float shadow = ndl > 0.0 ? sunShadow(vWorldRel, N) : 0.0;
    float skyAccess = skyL * skyL; // voxel sky light gates how much sun reaches here

    // warm strong sun vs cool dim sky ambient — the contrast is what keeps the scene
    // from going milky once tonemapped
    vec3 direct = vec3(2.05, 1.62, 1.08) * (ndl * shadow * skyAccess);
    vec3 ambient = vec3(0.26, 0.36, 0.55) * (0.10 + 0.90 * skyL);
    vec3 torch = vec3(1.35, 0.95, 0.55) * (blockL * blockL * 1.6);

    outColor = vec4(vAlbedo * kShade[vDir] * (direct + ambient + torch), 1.0);
}
