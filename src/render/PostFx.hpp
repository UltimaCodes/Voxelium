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
    bool dayCycle = true;  // advance time of day automatically
    float timeOfDay = 0.32f; // 0 midnight, 0.25 sunrise, 0.5 noon, 0.75 sunset
    float dayLengthSec = 240.0f; // real seconds for a full day
    float exposure = 1.0f;
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
        const ShadowMap* shadows;
        FxToggles toggles;
        float time = 0.0f;       // drives cloud drift
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
    unsigned int m_progTaa = 0, m_progSsgi = 0;
    unsigned int m_vao = 0;

    unsigned int m_hdrFbo = 0, m_hdrColor = 0, m_depth = 0;
    unsigned int m_sceneCopy = 0; // opaque snapshot for water SSR
    unsigned int m_hdr2Fbo = 0, m_hdr2 = 0;

    // TAA history ping-pong: read prev, write cur, flip. m_taaValid gates the first frame.
    unsigned int m_taa[2] = {0, 0}, m_taaFbo[2] = {0, 0};
    int m_taaIndex = 0;
    bool m_taaValid = false;
    unsigned int m_ssaoFbo = 0, m_ssaoTex = 0;
    unsigned int m_ssaoBlurFbo = 0, m_ssaoBlurTex = 0;
    unsigned int m_ssgiFbo = 0, m_ssgiTex = 0; // half-res RGBA16F bounce

    static constexpr int kBloomMips = 5;
    unsigned int m_bloomFbo[kBloomMips] = {};
    unsigned int m_bloomTex[kBloomMips] = {};
    glm::ivec2 m_bloomSize[kBloomMips] = {};

    int m_width = 0, m_height = 0;
};

} // namespace vx
