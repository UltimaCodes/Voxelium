#pragma once

#include <glm/glm.hpp>

#include <filesystem>

namespace vx {

class ShadowMap;

struct FxToggles {
    bool shadows = true;
    bool ssao = true;
    bool volumetric = true;
    bool bloom = true;
    bool clouds = true;
    bool ssgi = true;      // screen-space one-bounce indirect
    bool taa = false;      // enabled by the live app; off keeps offscreen/smoke renders exact
    bool motionBlur = false; // camera motion blur (uses TAA reprojection data)
    bool fog = true;         // distance fog into the sky at the render edge
    bool dayCycle = true;  // advance time of day automatically
    float timeOfDay = 0.32f; // 0 midnight, 0.25 sunrise, 0.5 noon, 0.75 sunset
    float dayLengthSec = 240.0f; // real seconds for a full day
    float exposure = 1.0f;
    float bloomStrength = 0.55f;
    // Self-illumination for emitter faces (lamps, glow). Tuned against the lamp palette:
    // the saturated swatches sit around 0.9 albedo on their brightest channel, so 2.0 puts
    // them clearly over the bloom threshold (1.15) while the pale ones still bloom softly
    // rather than blowing out. 0 turns emitters back into ordinary lit blocks.
    float emissive = 2.0f;
    // Grade. Neutral defaults on purpose: these are a look control, and the reference
    // images the harnesses compare against should not move because someone shipped a
    // fashionable teal-and-orange default.
    float saturation = 1.0f;   // 0 = greyscale, 1 = untouched
    float contrast = 1.0f;     // 1 = untouched
    float temperature = 0.0f;  // -1 cool .. +1 warm
    float vignette = 0.22f;    // 0 = off; subtle by default, it frames the shot
    // Animated, so it is the one effect that makes consecutive frames differ. Off by
    // default and left off by --smoke/--bench/--tour, for the same reason TAA is.
    float grain = 0.0f;
    // Eye adaptation. Its state is one texel carried across frames, so the image depends on
    // how many frames were rendered before it — the offscreen harnesses turn it off so
    // their reference images stay comparable, exactly as they do for TAA.
    bool autoExposure = true;
    // Lens flare strength. Screen-space, gated on the sun actually being visible.
    float lensFlare = 0.5f;
    // Cloud shadows on terrain. Shares the sky's density field, so it only means anything
    // with clouds on; gated separately because it costs a per-fragment fbm.
    float cloudShadow = 0.6f;
    // crosshair tint (RGBA, straight alpha). not a post effect, but it rides along in the
    // same struct so the settings menu and the settings file only have one thing to carry.
    float crosshair[4] = {1.0f, 1.0f, 1.0f, 0.85f};
    // Which lamp swatch to place, as an index into blocks::kLampRGB. An index rather than
    // an rgb triple because these are the only colours that exist — a lamp's colour is its
    // block id — so storing a free colour would only invite snapping it back again. Set by
    // the picker that opens on middle-clicking a lamp, and reused for everything placed
    // after that.
    int lampSwatch = 1;
    float volume = 0.7f; // master audio volume, same reason as crosshair: one struct
    // Internal render resolution as a fraction of the window. The whole HDR chain (scene,
    // ssao, ssgi, volumetrics, bloom, taa) runs at this size and only the final tonemap
    // pass runs at native, so it scales nearly everything that costs fill rate. 1.0 = off.
    float renderScale = 1.0f;
};

// The HDR post stack: scene renders into an RGBA16F target with a depth texture, then
// SSAO (half res) -> composite (AO apply + volumetric scattering) -> bloom
// downsample/upsample chain -> ACES + sRGB into whatever framebuffer is bound.
class PostFx {
public:
    bool init(const std::filesystem::path& shaderDir);
    void shutdown();
    void resize(int width, int height);

    void beginScene(); // binds the HDR framebuffer; caller clears and draws the world

    // Snapshot the opaque HDR color so the water pass can read it for screen-space
    // reflections/refraction, then rebind the HDR framebuffer to draw water into.
    void captureOpaqueForWater();
    unsigned int sceneCopyTex() const { return m_sceneCopy; }
    unsigned int depthTex() const { return m_depth; }
    int width() const { return m_width; }
    int height() const { return m_height; }

    struct RunParams {
        glm::vec2 projScale; // (aspect/f, 1/f) for depth reconstruction
        float nearPlane;
        glm::mat3 invViewRot;
        glm::vec3 sunL;
        glm::vec3 sunColor{1.0f}; // day/night sun radiance (warm/dim near horizon, ~0 at night)
        glm::vec2 fog{1e9f, 1e9f}; // distance fog (start, end); huge = off
        const ShadowMap* shadows;
        FxToggles toggles;
        glm::vec3 camPos{0.0f}; // camera world position; clouds are world-anchored
        float time = 0.0f;       // drives cloud drift and grain
        float dt = 0.0f;         // seconds since the last frame; drives eye adaptation
        bool underwater = false; // camera eye submerged -> composite adds depth fog + tint
        // TAA reprojection: last frame's unjittered view-proj and how far the camera moved
        glm::mat4 prevViewProj{1.0f};
        glm::vec3 camDelta{0.0f};
        bool taaHistoryValid = false;
        unsigned int targetFbo; // 0 = backbuffer
        int targetWidth, targetHeight;
    };
    void run(const RunParams& p);

    // TAA history is stale after a resize or when TAA was off last frame.
    void invalidateTaaHistory() { m_taaValid = false; }

private:
    void createTargets(int width, int height);
    void destroyTargets();

    unsigned int m_progSsao = 0, m_progBlur = 0, m_progComposite = 0;
    unsigned int m_progBloomDown = 0, m_progBloomUp = 0, m_progFinal = 0;
    unsigned int m_progTaa = 0, m_progSsgi = 0, m_progMotion = 0;
    unsigned int m_progLumDown = 0, m_progExpose = 0;
    unsigned int m_vao = 0;

    unsigned int m_hdrFbo = 0, m_hdrColor = 0, m_depth = 0;
    unsigned int m_sceneCopy = 0; // opaque snapshot for water SSR
    unsigned int m_hdr2Fbo = 0, m_hdr2 = 0;
    unsigned int m_motionFbo = 0, m_motionTex = 0; // camera motion blur output

    // TAA history ping-pong: read prev, write cur, flip. m_taaValid gates the first frame.
    unsigned int m_taa[2] = {0, 0}, m_taaFbo[2] = {0, 0};
    int m_taaIndex = 0;
    bool m_taaValid = false;
    unsigned int m_ssaoFbo = 0, m_ssaoTex = 0;
    unsigned int m_ssaoBlurFbo = 0, m_ssaoBlurTex = 0;
    unsigned int m_ssgiFbo = 0, m_ssgiTex = 0; // half-res RGBA16F bounce

    // Auto-exposure. kLumSize must be a power of two: the average is the last mip, so the
    // chain has to bottom out at exactly 1x1.
    static constexpr int kLumSize = 64;
    static constexpr int kLumMips = 7; // log2(64) + 1
    unsigned int m_lumTex = 0, m_lumFbo = 0;
    unsigned int m_adapt[2] = {0, 0}, m_adaptFbo[2] = {0, 0};
    int m_adaptIndex = 0;
    bool m_adaptValid = false;

    static constexpr int kBloomMips = 5;
    unsigned int m_bloomFbo[kBloomMips] = {};
    unsigned int m_bloomTex[kBloomMips] = {};
    glm::ivec2 m_bloomSize[kBloomMips] = {};

    int m_width = 0, m_height = 0;
};

} // namespace vx
