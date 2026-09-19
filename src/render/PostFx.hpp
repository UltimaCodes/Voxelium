#pragma once

#include <glm/glm.hpp>

#include <filesystem>
#include <string_view>

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
    // Grade. Was fully neutral on purpose, to keep the harnesses' reference images stable
    // against nobody having opinions about a "fashionable" default. That's still the reason
    // saturation stays untouched, but the owner's own reference images (shader-pack style:
    // warm highlights, a touch more contrast) are now an explicit ask, not a drive-by
    // preference (#78) - a SUBTLE lift, not a stylised grade, and --smoke/--bench/--tour all
    // inherit it same as a fresh player would, which is the point: the default look, not a
    // slider someone has to find first.
    float saturation = 1.0f;   // 0 = greyscale, 1 = untouched
    float contrast = 1.06f;    // 1 = untouched
    float temperature = 0.05f; // -1 cool .. +1 warm
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
    // Night rod-vision shift: desaturate and cool the darkest parts of the frame after dusk.
    // Faded in with darkness, so a day frame is untouched and the offscreen day references
    // do not move.
    float scotopic = 0.7f;
    // Cloud shadows on terrain. Shares the sky's density field, so it only means anything
    // with clouds on; gated separately because it costs a per-fragment fbm.
    float cloudShadow = 0.6f;
    // Leaf translucency: light coming THROUGH a backlit canopy rather than off it.
    float leafSss = 0.8f;
    // Share of a leaf canopy that lets sun straight through to the ground, as columns of
    // light rather than per-block holes (see shadow.frag). 0 = only the texture's own holes.
    float canopyLight = 0.22f;
    // Surface relief derived from the albedo's luminance — no authored normal maps, and no
    // room in the quad for tangent data. Costs four texture samples per opaque fragment.
    float relief = 0.6f;
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
    // Mixer buses. One flat volume also turned down the menu clicks and the wind bed along
    // with the world, which is why there are three.
    float volWorld = 1.0f;
    float volUi = 0.8f;
    float volAmbient = 0.6f;
    // Muffle a sound whose path to the listener is blocked by solid voxels. Costs one
    // raycast per sound EVENT (not per frame), against the same DDA the crosshair runs.
    float audioOcclusion = 0.75f;
    // Output dither, in 8-bit code values. Kills the banding in the sky gradient.
    float dither = 1.0f;
    // Weather (#29). `precip` is how hard it is coming down, `snow` picks rain vs snow.
    // Wetness is derived from precip rather than being its own knob: a surface that is wet
    // without visible rain, or dry in a downpour, is the sort of inconsistency nobody can
    // put their finger on but everybody notices.
    float precip = 0.0f;
    float snow = 0.0f;
    // Let the weather change on its own (render/WeatherDirector). precip and snow are then
    // driven, and the sliders only apply with this off.
    bool autoWeather = true;
    // Runtime only, never saved: filled in each frame by whatever is directing the sky.
    float cloudBias = 0.0f; // extra cloud cover, 0..~0.22
    float mist = 0.0f;      // 0..1, pulls the fog in (dawn, over water)
    // Shader quality tier, 0 Low .. 3 Ultra. Injected as VX_QUALITY and baked into the
    // programs, so changing it costs a shader rebuild — not a per-frame branch. 2 is the
    // build that existed before presets, which is what every other tier is judged against.
    int quality = 2;
    // Shadow cascade texture size. A resource knob: changing it reallocates the depth array
    // and is also injected, since the shaders divide by it to size their tap offsets.
    int shadowRes = 1536;
    // Internal render resolution as a fraction of the window. The whole HDR chain (scene,
    // ssao, ssgi, volumetrics, bloom, taa) runs at this size and only the final tonemap
    // pass runs at native, so it scales nearly everything that costs fill rate. 1.0 = off.
    float renderScale = 1.0f;
};

// The HDR post stack: scene renders into an RGBA16F target with a depth texture, then
// SSAO (half res) -> AO applied to the opaque scene (applyOpaqueSsao, before water exists in
// the buffer) -> water -> composite (volumetric scattering, SSGI) -> bloom
// downsample/upsample chain -> ACES + sRGB into whatever framebuffer is bound.
class PostFx {
public:
    bool init(const std::filesystem::path& shaderDir, std::string_view defines = {});
    void shutdown();
    // Rebuild every program at a new quality tier. Costs a compile the first time a tier is
    // seen and a cache hit thereafter — the binary cache keys on the injected source, so the
    // second visit to a preset is nearly free (#15 measured the whole set at 349 ms cold,
    // 56 ms warm). Returns false and leaves the old programs in place if anything fails, so
    // a bad tier cannot leave the renderer with no shaders.
    bool reloadPrograms(std::string_view defines);
    void resize(int width, int height);

    void beginScene(); // binds the HDR framebuffer; caller clears and draws the world

    // Snapshot the opaque HDR color so the water pass can read it for screen-space
    // reflections/refraction, then rebind the HDR framebuffer to draw water into.
    void captureOpaqueForWater();
    // Computes SSAO from the current (opaque-only) depth buffer and multiplies it onto the
    // opaque scene right now - call this AFTER captureOpaqueForWater() and BEFORE water is
    // drawn. Water never writes depth, so SSAO computed later (the old approach, once part
    // of the end-of-frame composite step) would be reading the LAKE BED's geometry under any
    // water on screen and darkening the water surface by however occluded the bed is - see
    // apply_ao.frag. Uses the snapshot captureOpaqueForWater() just took as the emitter-check
    // input, so it must run after that call, not before.
    void applyOpaqueSsao(bool enabled, const glm::vec2& projScale, float nearPlane);

    // Re-renders the cloud coverage texture around the camera. Call before the opaque pass;
    // terrain samples it (unit 6) for cloud shadows. Origin is snapped to whole texels so
    // the sample grid never shimmers as the camera moves.
    void updateCloudMap(const glm::dvec3& camPos, float time, float cloudBias);
    unsigned int cloudMapTex() const { return m_cloudTex; }
    // xy = world XZ of the map's corner, z = 1 / span
    glm::vec3 cloudMapXf() const { return {m_cloudOrigin.x, m_cloudOrigin.y, 1.0f / kCloudMapSpan}; }

    unsigned int sceneCopyTex() const { return m_sceneCopy; }
    unsigned int depthTex() const { return m_depth; }
    int width() const { return m_width; }
    int height() const { return m_height; }

    struct RunParams {
        glm::vec2 projScale; // (aspect/f, 1/f) for depth reconstruction
        float nearPlane;
        glm::mat3 invViewRot;
        glm::vec3 sunL;           // toward the TRUE sun: sky gradient, sun disc, lens flare
        // Toward whatever is lighting the world — sun by day, moon by night — and its
        // radiance. Volumetrics, contact shadows and caustics march against the cascades,
        // which are rendered from this direction, so they must use it and not sunL.
        glm::vec3 lightL{0.0f, 1.0f, 0.0f};
        glm::vec3 lightColor{1.0f};
        glm::vec3 sunColor{1.0f}; // the SUN's own radiance; exactly 0 at night
        float daylight = 1.0f;    // 1 full day, 0 after dusk
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

    unsigned int m_progSsao = 0, m_progBlur = 0, m_progComposite = 0, m_progApplyAo = 0;
    unsigned int m_progBloomDown = 0, m_progBloomUp = 0, m_progFinal = 0;
    unsigned int m_progTaa = 0, m_progSsgi = 0, m_progMotion = 0;
    unsigned int m_progLumDown = 0, m_progExpose = 0, m_progCloudMap = 0;
    unsigned int m_vao = 0;

    // 6 blocks a texel: the field's finest surviving octave is ~150 blocks across, so
    // nothing finer than that is worth storing.
    static constexpr int kCloudMapRes = 512;
    static constexpr float kCloudMapSpan = 3072.0f;
    unsigned int m_cloudTex = 0, m_cloudFbo = 0;
    glm::vec2 m_cloudOrigin{0.0f};

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

    std::filesystem::path m_shaderDir; // kept so a preset change can rebuild
    int m_width = 0, m_height = 0;
};

} // namespace vx
