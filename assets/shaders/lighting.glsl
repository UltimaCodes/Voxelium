// One light model for everything on land, so a hill looks the same near and far. The includer
// declares uCloudMap and uCloudMapXf and includes clouds.glsl first.

// Sky light in the open is 1, falls off smoothly, and a sealed cave reaches nothing. Down to
// level 11 (a forest floor, the shade of an overhang) it is the plain smoothstep. Below that it
// is light that has turned corners into a cave, and each turn takes most of it, so a tunnel goes
// from the twilight at its mouth to black within a dozen blocks instead of staying evenly grey.
float skyLevel(float s) {
    float into = smoothstep(0.0, 0.75, s);
    return s * s * (3.0 - 2.0 * s) * into * sqrt(into);
}
// The sun stops at a cave mouth well before the sky's own light does.
float sunGate(float s) { return smoothstep(0.2, 0.6, s); }

// Sky from above, warm bounce from the ground below.
vec3 skyFill(vec3 N, vec3 amb) { return mix(amb * vec3(0.66, 0.58, 0.46), amb, 0.5 + 0.5 * N.y); }

// Skips the shadow test, so faces turned from the sun or standing in shade keep their shape.
vec3 sunFill(vec3 N, vec3 L, vec3 sun) { return sun * (0.11 * (0.5 + 0.5 * dot(N, L))); }

// Added in squared space: a lamp in daylight does not blow out and dark corners keep their shape.
vec3 addLight(vec3 scene, vec3 torch, vec3 floorLight) {
    return sqrt(scene * scene + torch * torch + floorLight * floorLight);
}

const float kCloudMaxBlock = 0.5;

// 0 clear, 1 under thick cloud. A fixed sample height, so hills show no rings.
float cloudCover(vec3 worldPos, vec3 L) {
    if (L.y <= 0.05 || worldPos.y >= kCloudBase)
        return 0.0;
    vec2 uv = (worldPos.xz + L.xz * ((kCloudMid - 64.0) / L.y) - uCloudMapXf.xy) * uCloudMapXf.z;
    float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.2, 0.7, textureLod(uCloudMap, uv, 2.0).r) * smoothstep(0.0, 0.08, edge);
}

// What a climate does to green: lush and deep when wet, straw when dry, blue-grey when cold. A
// multiplier on the linear tile colour. Its far twin is in FarTerrain::build.
vec3 climateTint(float moisture, float temp) {
    float dry = smoothstep(0.45, 0.2, moisture), wet = smoothstep(0.5, 0.85, moisture);
    float cold = smoothstep(0.42, 0.22, temp);
    vec3 tint = mix(mix(vec3(1.0), vec3(0.60, 0.85, 0.42), wet), vec3(2.1, 1.22, 1.15), dry);
    return mix(tint, vec3(0.74, 0.76, 1.5), cold * 0.8);
}
