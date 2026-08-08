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

struct SkyLight {
    glm::vec3 dir;     // toward the sun/moon
    glm::vec3 color;   // direct radiance: warm+dim near the horizon, ~0 at night
    glm::vec3 ambient; // sky fill: blue by day, dark cool at night
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
