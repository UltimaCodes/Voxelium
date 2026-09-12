#version 460 core

// Injected by Shader.cpp from the active quality preset (see submitProgram).
//
// VX_QUALITY picks the sample and step counts below. They MUST stay compile-time constants:
// a uniform loop bound does not unroll, and a quality system whose top setting is slower
// than having no quality system at all is the wrong trade. Default 2 = High, the build these
// numbers were tuned in, so an un-injected compile matches the one before presets existed.
//
// VX_SHADOW_RES must agree with the texture the cascades were actually allocated at, which
// is why it comes through the same channel. A define rather than a uniform because every use
// is a divisor sizing a tap offset, and a compile-time divisor folds away.
#ifndef VX_QUALITY
#define VX_QUALITY 2
#endif
#ifndef VX_SHADOW_RES
#define VX_SHADOW_RES 1536.0
#endif

// applies SSAO to the scene, paints a real sky for pixels with no geometry (gradient +
// sun disc + horizon glow — no extra pass), and adds volumetric sun scattering
// raymarched through the shadow cascades for geometry pixels. output stays HDR; bloom
// and tonemapping come after.

in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D uScene;
layout(binding = 3) uniform sampler2D uDepth;
layout(binding = 4) uniform sampler2D uSsgi;
layout(binding = 1) uniform sampler2DArrayShadow uShadowMap;

layout(location = 0) uniform vec2 uProjScale; // (aspect/f, 1/f)
layout(location = 1) uniform float uNear;
layout(location = 2) uniform mat3 uInvViewRot; // view -> camera-relative world; eats 2,3,4
// The TRUE sun, even at night when it is below the horizon. The sky pass needs it to place
// the disc, aim the warm bands and put the moon opposite. For anything that SHADES, use
// uLightL — after dusk that points at the moon and the cascades are rendered from it.
layout(location = 5) uniform vec3 uSunL;
layout(location = 6) uniform mat4 uCascadeVp[3];
layout(location = 18) uniform vec3 uCascadeSplit;
layout(location = 19) uniform float uUnderwater; // 1 when the camera eye is submerged
layout(location = 20) uniform float uCloudsOn;
layout(location = 21) uniform float uTime;
layout(location = 22) uniform float uSsgiOn;
// location 23 (uAoOn) retired - AO moved to apply_ao.frag, run before water is drawn
layout(location = 24) uniform float uVolumetricOn; // moved off 4
layout(location = 25) uniform vec3 uSunColor;      // day/night sun radiance
layout(location = 26) uniform vec2 uFog;           // (start, end) distance; huge = off
layout(location = 30) uniform vec3 uLightL;        // toward sun by day, moon by night
layout(location = 31) uniform vec3 uLightColor;    // radiance of uLightL
layout(location = 32) uniform float uDaylight;     // 1 full day, 0 after dusk
layout(location = 27) uniform float uFlare;        // lens flare strength, 0 = off
layout(location = 28) uniform vec3 uCamPos;        // camera world position (clouds are world-anchored)
layout(location = 29) uniform float uContact;      // screen-space contact shadow, 0 = off

#include "clouds.glsl"

const vec3 kWaterDeep = vec3(0.03, 0.13, 0.20);
const float kSeaLevel = 62.0; // matches gen/TerrainGenerator kSeaLevel

// Caustics. Voronoi-ish: two layers of cellular distance, scrolled against each other and
// sharpened, which is the standard trick for the bright wandering web on a seabed. Two
// layers at different scales and speeds because ONE layer reads as an obvious repeating
// pattern the moment it covers any area — hiding the repeat is most of the work here.
float causticLayer(vec2 p, float t) {
    vec2 i = floor(p);
    vec2 f = p - i;
    float best = 1e9;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            vec2 g = vec2(float(x), float(y));
            // per-cell feature point, orbiting so the web moves rather than pulses
            vec2 o = vec2(cl_hash(i + g), cl_hash(i + g + 37.1));
            o = 0.5 + 0.5 * sin(t * 0.9 + 6.2831 * o);
            best = min(best, length(g + o - f));
        }
    return best;
}

// 0..1 caustic brightness at a world XZ, for a surface `depth` below the waterline.
float caustics(vec2 worldXz, float depth, float t) {
    vec2 p = worldXz * 0.55;
    float a = causticLayer(p, t);
    float b = causticLayer(p * 1.83 + 11.3, t * 1.17);
    // Sharpen: caustics are thin bright filaments between dark cells, not soft blobs.
    // CLAMP FIRST. causticLayer returns a cell distance, and a feature point in a diagonal
    // neighbour can sit ~2.1 away, so 1 - min(a,b) goes negative — and pow() with a negative
    // base is undefined in GLSL. This driver hands back NaN, which propagates through the
    // tonemap and comes out as solid black blocks across the seabed.
    float c = pow(clamp(1.0 - min(a, b), 0.0, 1.0), 6.0);
    // They are strongest just under the surface and wash out with depth as the light
    // spreads, and they cannot exist above the waterline at all.
    return c * exp(-depth * 0.16);
}

float ign(vec2 p); // defined below; the cloud march dithers with it

vec3 viewRay() {
    vec2 ndc = vUv * 2.0 - 1.0;
    return normalize(uInvViewRot * vec3(ndc.x * uProjScale.x, ndc.y * uProjScale.y, -1.0));
}

float starHash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// Cube-face parameterisation of a direction: pick the dominant axis and divide by it, so
// every face gets an evenly spaced 2D grid. Folding a 3D direction into 2D any other way
// (e.g. dir.xy + dir.z) makes whole families of directions share a cell, and the stars
// come out as streaks instead of points.
vec2 cubeFaceUv(vec3 dir, out float face) {
    vec3 a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) {
        face = dir.x > 0.0 ? 0.0 : 1.0;
        return dir.zy / a.x;
    }
    if (a.y >= a.z) {
        face = dir.y > 0.0 ? 2.0 : 3.0;
        return dir.xz / a.y;
    }
    face = dir.z > 0.0 ? 4.0 : 5.0;
    return dir.xy / a.z;
}

// One star per grid cell, placed at a random point inside it and drawn as a round dot
// with a soft edge. Most cells are empty; brightness and colour vary so the field reads
// as a sky rather than a regular dot screen.
vec3 starField(vec3 dir, float night) {
    float face;
    vec2 uv = cubeFaceUv(dir, face) * 90.0;
    vec2 cell = floor(uv);
    vec2 f = uv - cell;
    vec2 key = cell + face * 311.0;

    float pick = starHash(key);
    if (pick < 0.960)
        return vec3(0.0);

    vec2 centre = vec2(starHash(key + 11.3), starHash(key + 23.7)) * 0.7 + 0.15;
    float d = length(f - centre);
    float mag = starHash(key + 41.1);            // most stars faint, a few bright
    float radius = mix(0.055, 0.13, mag * mag);
    float dot0 = smoothstep(radius, 0.0, d);
    // slow, per-star twinkle
    float tw = 0.72 + 0.28 * sin(uTime * 1.7 + pick * 90.0);
    // colour: cool white through faint gold
    vec3 tint = mix(vec3(0.78, 0.85, 1.0), vec3(1.0, 0.92, 0.78), starHash(key + 57.9));
    return tint * dot0 * tw * (0.55 + 1.35 * mag * mag) * night;
}

// smooth value noise — the blocky floor() cells of a raw hash were visible as squares
// across something as small on screen as the moon
float vnoise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(starHash(i), starHash(i + vec2(1.0, 0.0)), f.x),
               mix(starHash(i + vec2(0.0, 1.0)), starHash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// A moon that reads as a sphere, not a white disc: cratered surface, curvature shading
// toward the limb, and a soft halo. Kept deliberately dim in HDR — pushed much above 1
// the tonemapper crushes the whole surface to flat white, which is what made it look
// like a small sun.
vec3 moonDisc(vec3 dir, vec3 moonDir, float night) {
    const float kRadius = 0.030; // angular radius
    float md = dot(dir, moonDir);
    if (md <= 0.0)
        return vec3(0.0);

    vec3 mx = normalize(cross(moonDir, vec3(0.0, 1.0, 0.0)) + vec3(1e-4, 0.0, 0.0));
    vec3 my = cross(mx, moonDir);
    vec2 mc = vec2(dot(dir, mx), dot(dir, my)) / kRadius; // -1..1 across the face
    float r = length(mc);

    vec3 halo = vec3(0.22, 0.26, 0.38) * pow(max(md, 0.0), 1400.0) * night * 0.7;
    if (r >= 1.0)
        return halo;

    float limb = sqrt(max(1.0 - r * r, 0.0)); // sphere normal's z: darkens toward the edge
    // maria + craters, a few octaves over the face
    float m = vnoise2(mc * 2.1 + 3.0) * 0.55 + vnoise2(mc * 5.3 + 11.0) * 0.30 +
              vnoise2(mc * 12.0 + 23.0) * 0.15;
    vec3 surface =
        mix(vec3(0.34, 0.36, 0.42), vec3(0.88, 0.88, 0.83), smoothstep(0.30, 0.78, m));
    float edge = smoothstep(1.0, 0.93, r); // antialias the silhouette
    return (surface * (0.45 + 0.55 * limb) * night * edge) + halo;
}

vec3 skyColor(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    float alt = uSunL.y;                             // sun altitude, -1..1
    float day = clamp(alt * 3.0 + 0.12, 0.0, 1.0);   // 1 day, 0 night
    // day/night gradient
    // Night endpoints cut ~2.5x (was 0.01/0.015/0.05 zenith, 0.04/0.06/0.12 horizon). The
    // measurement that forced this: at the --tour night stop the SKY is most of the frame's
    // brightness, and distance fog melts the far terrain into it too, so the sky gradient
    // was setting the exposure of the whole image. A horizon green of 0.06 lands at ~70/255
    // after ACES and the gamma encode, which is dusk, not night — and no amount of retuning
    // the surface shading can pull an image down past what its own sky is sitting at.
    // Stars and the moon are ADDED after this, so darkening here makes them read stronger.
    vec3 zenith = mix(vec3(0.004, 0.006, 0.020), vec3(0.14, 0.34, 0.80), day);
    vec3 horizon = mix(vec3(0.016, 0.025, 0.050), vec3(0.66, 0.78, 0.92), day);
    vec3 col = mix(horizon, zenith, pow(up, 0.5));
    if (dir.y < 0.0)
        col = mix(horizon, mix(vec3(0.024, 0.028, 0.036), vec3(0.34, 0.38, 0.44), day),
                  clamp(-dir.y * 3.0, 0.0, 1.0));

    float sd = max(dot(dir, uSunL), 0.0);
    float above = smoothstep(-0.03, 0.06, alt);
    float lowSun = clamp(1.0 - abs(alt) * 2.2, 0.0, 1.0) * day; // sunrise/sunset warmth
    col += vec3(1.0, 0.34, 0.10) * pow(sd, 3.0) * lowSun * 2.4;            // warm band
    col = mix(col, col * vec3(1.2, 0.92, 0.72), lowSun * 0.5);            // golden-hour cast
    col += vec3(1.0, 0.55, 0.28) * pow(sd, 8.0) * 0.6 * above;             // glow
    col += uSunColor * smoothstep(0.9986, 0.9995, sd) * 3.2 * above;       // sun disc

    // Stars fade out well before the sky is properly light — a linear 1-day left them
    // visible as bright specks through dawn and dusk.
    float night = pow(1.0 - day, 2.2);
    if (night > 0.004) {
        col += starField(dir, night) * step(-0.02, dir.y); // none below the horizon
        col += moonDisc(dir, -uSunL, night);               // moon rides opposite the sun
    }
    return col;
}

// Raymarched clouds. The field itself lives in clouds.glsl because the terrain shader
// samples the same one for shadows.
//
// Marched, not painted: the ray is intersected with the slab and stepped through it,
// accumulating extinction with Beer's law and lighting each step with a short march toward
// the sun. That is what gives silver lining, depth at the edges, and clouds that pass
// overhead instead of sliding with the camera.
vec4 clouds(vec3 dir) {
    if (dir.y < 0.02)
        return vec4(0.0); // looking at or below the horizon: the slab is never hit

    float t0 = (kCloudBase - uCamPos.y) / dir.y;
    float t1 = (kCloudTop - uCamPos.y) / dir.y;
    if (t1 < t0) { float tmp = t0; t0 = t1; t1 = tmp; } // inside or above the layer
    t0 = max(t0, 0.0);
    if (t1 <= t0)
        return vec4(0.0);
    // Cap the marched span: looking near the horizon the slab is kilometres long, and the
    // far end contributes nothing but cost.
    t1 = min(t1, t0 + 2600.0);

#if   VX_QUALITY == 0
    const int kSteps = 6;
#elif VX_QUALITY == 1
    const int kSteps = 8;
#elif VX_QUALITY == 3
    const int kSteps = 20;
#else
    const int kSteps = 12;
#endif
    float stepLen = (t1 - t0) / float(kSteps);
    // Dither the start so the step boundaries do not band across the sky.
    float jitter = ign(gl_FragCoord.xy) * stepLen;

    float transmittance = 1.0;
    vec3 scattered = vec3(0.0);
    float day = uDaylight;
    vec3 sunTint = mix(vec3(0.55, 0.60, 0.70), vec3(1.05, 1.02, 0.98), day);

    for (int i = 0; i < kSteps; ++i) {
        float t = t0 + jitter + stepLen * float(i);
        vec3 p = uCamPos + dir * t;
        float d = cloudDensity(p, uTime);
        if (d <= 0.001)
            continue;
        // Self-shadowing: three coarse steps toward the sun. Cheap, and it is what separates
        // a lit top from a heavy base.
        float shadowD = 0.0;
        for (int j = 1; j <= 3; ++j)
            shadowD += cloudDensity(p + uLightL * (float(j) * 26.0), uTime);
        float lit = exp(-shadowD * 0.55);
        // Forward scattering: bright rim when looking toward the sun through thin cloud.
        float phase = 0.55 + 0.45 * pow(max(dot(dir, uLightL), 0.0), 4.0);
        // The ambient half was an ungated constant, so a cloud deck stayed lit after dark
        // and the cloud band measured brighter than the sky it sat in. Faded with daylight,
        // keeping 12% so moonlit cloud still reads as shape rather than a hole in the stars.
        vec3 lightCol = uLightColor * lit * phase +
                        vec3(0.30, 0.36, 0.46) * 0.55 * mix(0.12, 1.0, uDaylight);
        float extinction = d * 0.055 * stepLen;
        float absorbed = 1.0 - exp(-extinction);
        scattered += transmittance * absorbed * lightCol * sunTint;
        transmittance *= 1.0 - absorbed;
        if (transmittance < 0.02)
            break;
    }
    float alpha = clamp(1.0 - transmittance, 0.0, 1.0);
    if (alpha <= 0.001)
        return vec4(0.0);
    // Fade the layer out toward the horizon, where the march is least accurate and clouds
    // would otherwise pile into a hard band.
    alpha *= smoothstep(0.02, 0.16, dir.y);
    return vec4(scattered / max(alpha, 1e-4), alpha);
}

vec3 skyWithClouds(vec3 dir) {
    vec3 s = skyColor(dir);
    if (uCloudsOn > 0.5) {
        vec4 c = clouds(dir);
        s = mix(s, c.rgb, c.a);
    }
    return s;
}

// 4-tap softened shadow lookup for the raymarch — a single tap makes the light shafts
// alias into a grid against the shadow texels; averaging a few taps smooths them out.
float shadowSingle(vec3 pos) {
    float d = length(pos);
    if (d >= uCascadeSplit.z)
        return 1.0;
    int c = d < uCascadeSplit.x ? 0 : (d < uCascadeSplit.y ? 1 : 2);
    vec4 sc = uCascadeVp[c] * vec4(pos, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    float z = sc.z - 0.0015;
    float t = 1.5 / VX_SHADOW_RES;
    return 0.25 * (texture(uShadowMap, vec4(uv + vec2(t, t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(-t, t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(t, -t), float(c), z)) +
                   texture(uShadowMap, vec4(uv + vec2(-t, -t), float(c), z)));
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// interleaved gradient noise — a much better dither than a plain hash, so the raymarch
// start offset breaks up banding evenly instead of in blotches
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

// Screen-space lens flare for the one light that matters. The ghosts are procedural rather
// than resampled scene colour: the resample trick wants a downsampled thresholded buffer to
// be affordable, and for a single dominant source a handful of tinted discs along the
// sun-through-centre axis is cheaper and far easier to keep tasteful.
//
// Added in HDR, before bloom, so the flare blooms like the sun does instead of sitting on
// top of the image as a decal.
vec3 lensFlare() {
    if (uFlare <= 0.0)
        return vec3(0.0);
    // uInvViewRot maps view -> camera-relative world, so its transpose is world -> view.
    vec3 v = transpose(uInvViewRot) * uSunL;
    if (v.z >= -1e-4)
        return vec3(0.0); // sun behind the camera
    vec2 sunUv = vec2((v.x / -v.z) / uProjScale.x, (v.y / -v.z) / uProjScale.y) * 0.5 + 0.5;
    if (any(lessThan(sunUv, vec2(-0.4))) || any(greaterThan(sunUv, vec2(1.4))))
        return vec3(0.0);

    // Occlusion, five taps. One tap makes the whole flare switch on and off between frames
    // as a leaf or a chimney crosses the sun; five lets it fade. Reversed-z: far = 0 = sky.
    const vec2 kTaps[5] = vec2[5](vec2(0.0), vec2(0.014, 0.0), vec2(-0.014, 0.0),
                                  vec2(0.0, 0.014), vec2(0.0, -0.014));
    float vis = 0.0;
    for (int i = 0; i < 5; ++i)
        vis += texture(uDepth, clamp(sunUv + kTaps[i], vec2(0.0), vec2(1.0))).r <= 0.0 ? 0.2
                                                                                      : 0.0;
    if (vis <= 0.0)
        return vec3(0.0);

    // Fade out as the sun leaves the frame — a flare that pops the instant the sun crosses
    // the edge is worse than no flare.
    float edge = 1.0 - smoothstep(0.40, 0.85, length(sunUv - 0.5));
    if (edge <= 0.0)
        return vec3(0.0);

    const float kPos[6] = float[6](0.32, 0.60, 0.95, 1.30, 1.70, 2.15);
    const float kSize[6] = float[6](0.060, 0.028, 0.085, 0.020, 0.048, 0.100);
    const vec3 kTint[6] =
        vec3[6](vec3(0.30, 0.50, 1.00), vec3(1.00, 0.55, 0.28), vec3(0.35, 1.00, 0.60),
                vec3(1.00, 0.32, 0.80), vec3(0.45, 0.70, 1.00), vec3(1.00, 0.85, 0.42));

    const float aspect = uProjScale.x / max(uProjScale.y, 1e-6);
    const vec2 axis = 0.5 - sunUv; // toward the centre, and out the far side
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 6; ++i) {
        vec2 q = (vUv - (sunUv + axis * kPos[i])) * vec2(aspect, 1.0);
        // 1 - smoothstep, NOT smoothstep(size, 0, r): a descending smoothstep has
        // edge0 > edge1, which is undefined in GLSL and returns 0 on this driver —
        // every ghost silently evaluated to nothing.
        sum += kTint[i] * (1.0 - smoothstep(0.0, kSize[i], length(q))) * 0.30;
    }
    // Halo: a soft ring about the mirrored sun, which is what sells it as glass rather than
    // as a row of coloured dots.
    {
        vec2 q = (vUv - (sunUv + axis * 2.0)) * vec2(aspect, 1.0);
        float r = length(q);
        sum += vec3(1.0, 0.72, 0.45) * (1.0 - smoothstep(0.0, 0.020, abs(r - 0.16))) * 0.22;
    }
    return sum * uSunColor * vis * edge * uFlare;
}

// Screen-space contact shadows. The cascades cannot resolve the small stuff — a tuft
// against the ground, a block's own lip, the join where a trunk meets the dirt — because
// one shadow texel spans several blocks out at the far split. This marches the depth buffer
// a short way toward the sun and darkens what it finds occluded.
//
// It has to live HERE rather than in chunk.frag: during the opaque pass the depth buffer is
// the thing being written, and sampling it there is a feedback loop. Composite is the first
// point where a complete depth buffer exists.
float contactShadow(vec3 worldRel, float rawD) {
    if (uContact <= 0.0)
        return 1.0;
#if   VX_QUALITY <= 1
    const int kSteps = 4;
#elif VX_QUALITY == 3
    const int kSteps = 12;
#else
    const int kSteps = 8;
#endif
    const float kReach = 1.6; // world units; beyond this the cascades already have it
    float occl = 0.0;
    // Dither the start so the fixed step count does not band along surfaces.
    float t0 = (0.15 + 0.85 * ign(gl_FragCoord.xy)) * (kReach / float(kSteps));
    for (int i = 0; i < kSteps; ++i) {
        float t = t0 + kReach * float(i) / float(kSteps);
        vec3 p = worldRel + uLightL * t;
        // project back to the screen: world -> view -> ndc, the inverse of the
        // reconstruction above
        vec3 v = transpose(uInvViewRot) * p;
        if (v.z >= -1e-4)
            continue;
        vec2 uv = vec2((v.x / -v.z) / uProjScale.x, (v.y / -v.z) / uProjScale.y) * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))
            continue;
        float sceneRaw = texture(uDepth, uv).r;
        if (sceneRaw <= 0.0)
            continue; // sky: nothing to be occluded by
        // reversed-z: bigger raw depth = nearer. The sample is behind geometry when the
        // scene there is nearer than the marched point.
        float sampleViewZ = v.z;
        float sceneViewZ = -uNear / max(sceneRaw, 1e-7);
        float diff = sceneViewZ - sampleViewZ; // >0 means the scene is in front
        // The thickness window stops a distant surface far behind the ray from counting as
        // a blocker, which is what makes naive versions shadow the whole background.
        if (diff > 0.02 && diff < 0.9) {
            // Falloff keyed to the actual marched distance t, not the step INDEX - with only
            // a handful of steps, 1.0 - i/kSteps quantises into a handful of discrete jumps
            // (coarser at low VX_QUALITY, where kSteps drops to 4), which read as banding on
            // top of an already-strong effect. t/kReach is continuous regardless of step
            // count, so lowering quality costs sample density, not smoothness.
            occl = max(occl, 1.0 - clamp(t / kReach, 0.0, 1.0));
        }
    }
    return 1.0 - uContact * occl;
}

void main() {
    float rawD = texture(uDepth, vUv).r;
    vec3 dir = viewRay();
    // Applies to sky and geometry alike: it is in the lens, not in the world.
    vec3 flare = uUnderwater > 0.5 ? vec3(0.0) : lensFlare();

    // sky pixel (reversed-z far = 0): paint the sky instead of the flat clear colour.
    // underwater there is no sky — the far distance is just deep water.
    if (rawD <= 0.0) {
        outColor = vec4((uUnderwater > 0.5 ? kWaterDeep : skyWithClouds(dir)) + flare, 1.0);
        return;
    }

    vec3 color = texture(uScene, vUv).rgb;
    // AO used to be applied right here, to the whole finished frame including water - moved
    // to apply_ao.frag, run on the opaque scene BEFORE water is drawn on top of it. See that
    // file for why: water never writes depth, so AO computed from the lake bed underneath it
    // was tracing the bed's own terrain contours onto the water surface above. The emitter
    // exception (a lamp's own light is not RECEIVED ambient light, so occluding it at its
    // base was a dark band under every lamp, #49) moved there too, using the same luminance
    // window (an emitter totals ~3.2 pre-tonemap, the brightest sunlit terrain ~1.7 - a
    // threshold of 1.0 leaked onto 2.4% of a daylight frame, 2.0 leaves 0).
    if (uSsgiOn > 0.5)
        color += texture(uSsgi, vUv).rgb * 1.15; // one-bounce indirect fill (#75 raised to 1.6,
        // walked back after "too bright, no contrast" - see ssgi.frag's kRadius comment)

    // reconstruct the camera-relative world position once, for volumetrics + fog
    float viewZ = -uNear / max(rawD, 1e-7);
    vec2 ndc = vUv * 2.0 - 1.0;
    vec3 worldRel = uInvViewRot *
                    vec3(ndc.x * uProjScale.x * -viewZ, ndc.y * uProjScale.y * -viewZ, viewZ);
    float dist = length(worldRel);

    // Only where the cascade says the sun reaches: darkening a pixel the sun never lit is
    // just AO, and SSAO has already had its turn.
    //
    // Scaled by how much sun there actually is. A contact shadow is a SUN shadow, but this
    // multiplies the composited colour rather than the direct term alone — so after dark,
    // when ambient is the only light left, it was darkening light the sun never cast. That
    // is what made night shadows read as far too heavy.
    float sunAmt = clamp(max(uLightColor.r, max(uLightColor.g, uLightColor.b)), 0.0, 1.0);
    if (uContact > 0.0 && sunAmt > 0.01 && shadowSingle(worldRel) > 0.5)
        color *= mix(1.0, contactShadow(worldRel, rawD), sunAmt);

    if (uVolumetricOn > 0.5) {
        float len = min(dist, 150.0);
        vec3 rd = normalize(worldRel);

#if   VX_QUALITY == 0
        const int kSteps = 6;
#elif VX_QUALITY == 1
        const int kSteps = 8;
#elif VX_QUALITY == 3
        const int kSteps = 24;
#else
        const int kSteps = 16;
#endif
        float stepLen = len / float(kSteps);
        float t = stepLen * ign(gl_FragCoord.xy); // dithered start, breaks up banding
        float scatter = 0.0;
        for (int i = 0; i < kSteps; ++i) {
            scatter += shadowSingle(rd * t);
            t += stepLen;
        }
        scatter /= float(kSteps);
        // stronger forward lobe + a touch more intensity so the god-ray shafts read
        // clearly when looking toward the sun, without bringing the aliased grid back
        float phase = 0.10 + 0.90 * pow(max(dot(rd, uLightL), 0.0), 5.0);
        // uSunColor carries the day/night intensity, so the shafts fade out at night.
        // Raised from 0.00018 (#76): reference shader-pack screenshots show god rays as a
        // dominant compositional element through gaps in foliage/terrain, not a faint haze.
        color += uLightColor * (scatter * phase * len * 0.00032);
    }

    if (uUnderwater > 0.5) {
        // Caustics on whatever the light lands on down here. Driven by world position, so
        // they stay put on the seabed as you swim rather than sliding with the view.
        vec3 wpos = uCamPos + worldRel;
        float depth = max(kSeaLevel - wpos.y, 0.0);
        float caus = caustics(wpos.xz, depth, uTime);
        // Only where the sun actually reaches: caustics are focused sunlight, so they have
        // no business on a face in shadow or in a cave.
        float lit = shadowSingle(worldRel) * uDaylight;
        color += uSunColor * caus * lit * 0.55;

        // underwater: absorb reds and fog toward deep water with distance (Beer-Lambert-ish)
        color *= vec3(0.55, 0.82, 1.0);
        color = mix(color, kWaterDeep, 1.0 - exp(-dist * 0.10));

        // Underwater god rays. The volumetric pass above marches the shadow cascades, which
        // gives shafts through gaps in the terrain; this adds the surface itself as the
        // occluder, so light comes down in bands broken by the swell. Cheap because it only
        // needs the ray's crossing of the waterline, not a march.
        if (uVolumetricOn > 0.5 && uSunL.y > 0.05 && uDaylight > 0.0) {
            vec3 rd = normalize(worldRel);
            float tSurf = (kSeaLevel - uCamPos.y) / max(rd.y, 1e-3);
            if (tSurf > 0.0 && tSurf < dist) {
                vec2 hit = (uCamPos + rd * tSurf).xz;
                float shaft = caustics(hit * 0.6, 0.0, uTime * 0.8);
                float fade = smoothstep(0.0, 40.0, tSurf) * (1.0 - smoothstep(40.0, 140.0, tSurf));
                color += uSunColor * shaft * fade * pow(max(dot(rd, uSunL), 0.0), 3.0) * 0.5;
            }
        }
    } else {
        // distance fog: melt the far edge of the loaded world into the sky, so the render
        // boundary reads as haze/horizon instead of a hard cut
        float fog = smoothstep(uFog.x, uFog.y, dist);
        if (fog > 0.001)
            color = mix(color, skyWithClouds(dir), fog);
    }

    outColor = vec4(color + flare, 1.0);
}
