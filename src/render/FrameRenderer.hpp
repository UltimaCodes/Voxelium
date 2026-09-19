#pragma once

#include "gfx/GpuTimer.hpp"
#include "render/Camera.hpp"
#include "render/Frustum.hpp"
#include "render/PostFx.hpp"

#include <glm/glm.hpp>

namespace vx {

class Renderer;
class ShadowMap;
class Overlay;
class Particles;
class Weather;
class Wildlife;

// Everything needed to draw one frame of the world, shared by the live game loop and by all
// three offscreen harnesses (--smoke, --bench, --tour). It lives here rather than in main so
// the harnesses render through EXACTLY the same path the game does — a harness with its own
// copy of the frame would eventually measure a frame nobody plays.

struct CameraFrame {
    glm::mat4 viewProj;
    Frustum frustum;
    glm::ivec3 cameraChunk;
    glm::vec3 cameraFrac;
};

CameraFrame buildCameraFrame(const FlyCamera& camera, int fbWidth, int fbHeight,
                             glm::vec2 jitterNdc = glm::vec2(0.0f));

// The two meanings of "the sun" that used to be one vector, now separated because they stop
// agreeing after dusk.
//
// `sunDir` is where the sun actually is. The sky pass needs it to place the disc, aim the
// warm horizon bands, and derive whether it is day at all — flipping it would render
// midnight as noon.
//
// `lightDir` is where light arrives FROM: the sun by day, the moon by night. Everything that
// shades a surface takes this one — shadow cascades, NdotL, specular, volumetrics, contact
// shadows. Because the moon is drawn at exactly -sunDir, the two are the same vector flipped,
// so the whole night lighting path costs one negation and no new shadow pass.
//
// The handover happens where `daylight` has already reached 0 and `moonlight` has not yet
// left it, so both terms are zero at the instant the direction flips and nothing pops.
struct SkyLight {
    glm::vec3 sunDir;   // toward the true sun, for the sky pass only
    glm::vec3 lightDir; // toward whatever is lighting the world right now
    glm::vec3 color;    // radiance of lightDir: warm sun by day, dim cool moon by night
    glm::vec3 sunColor; // radiance of the SUN specifically; 0 at night. Sun disc, lens flare
    glm::vec3 ambient;  // sky fill: blue by day, dark cool at night
    float daylight;     // 1 full day, 0 after dusk. Was derived in four shaders from sunL.y
};

// Sun along a day arc from time-of-day (0.25 sunrise east, 0.5 noon, 0.75 sunset west).
SkyLight computeSky(float timeOfDay);

// Low-discrepancy sub-pixel offset for the TAA jitter.
float halton(int i, int base);

// Carried across frames so the TAA resolve can reproject into the previous frame.
struct TaaState {
    glm::mat4 prevViewProj{1.0f};
    glm::dvec3 prevCamPos{0.0};
    int frame = 0;
    bool valid = false;
};

struct RenderContext {
    Renderer& renderer;
    ShadowMap& shadows;
    PostFx& postfx;
    Overlay& overlay;
    Particles* particles; // null in the offscreen paths that never spawn any
    Weather* weather;     // null where precipitation is not wanted
    Wildlife* wildlife;   // null where nothing lives; creatures are drawn before the water
    unsigned int drawProgram;
    unsigned int waterProgram;
    unsigned int blockTex;
};

struct FrameTimers {
    GpuTimer shadow, scene, post;
};

// One full frame: cascades -> HDR scene (opaque + transparent water) -> post into target.
void renderFrame(RenderContext& rc, const FlyCamera& camera, const FxToggles& fx,
                 bool caveCull, int fbW, int fbH, unsigned int targetFbo, float time,
                 FrameTimers* timers, bool hasSelection = false,
                 const glm::vec3& selMinCamRel = glm::vec3(0.0f), bool underwater = false,
                 TaaState* taa = nullptr, float fogEnd = 1e9f, int crackLayer = -1,
                 unsigned int blockTex = 0);

} // namespace vx
