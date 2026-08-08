#include "gfx/BlockTextures.hpp"

#include "core/Log.hpp"
#include "world/Chunk.hpp" // lamp hue table: the texture and the light it casts share it

#include <glad/gl.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <array>
#include <cmath>
#include <cstdint>

namespace vx {

namespace {

constexpr int kTile = 16;
constexpr int kCrackStages = 5;
constexpr int kNamedLayers = 15;
constexpr int kCrossLayers = 2; // grass tuft, flower
// the lamp hues tack onto the end, one layer each, generated from blocks::kLampRGB so
// the face you see and the light it casts can never drift apart
constexpr int kLampLayer0 = kNamedLayers;
constexpr int kLampLayerEnd = kNamedLayers + blocks::kLampCount;
// cross-plane foliage tiles: mostly transparent, so they are generated rather than tinted
constexpr int kCrossLayer0 = kLampLayerEnd;
// Break-progress crack stages, appended last so the block overlay can sample them out of
// the same array the world already has bound — no second texture, no second sampler.
constexpr int kCrackLayer0 = kLampLayerEnd + kCrossLayers;
constexpr int kLayers = kCrackLayer0 + kCrackStages;

// filename (without extension) per layer, looked up in assets/textures/
constexpr const char* kNames[kNamedLayers] = {"stone",    "dirt",     "grass_top", "grass_side",
                                         "glow",     "wood_side", "wood_top", "leaves",
                                         "sand",     "snow",     "planks",    "bricks",
                                         "coal_ore", "iron_ore", "gravel"};

// value hash in [0,1) — deterministic per texel so textures are identical every run
float h01(int x, int y, int salt) {
    std::uint32_t h = std::uint32_t(x) * 0x8DA6B343u ^ std::uint32_t(y) * 0xD8163841u ^
                      std::uint32_t(salt) * 0xCB1AB31Fu;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return float(h & 0xFFFFFFu) / float(0x1000000u);
}

struct Rgba {
    std::uint8_t r, g, b, a;
};

Rgba shade(float r, float g, float b, float m) {
    const auto c = [](float v) {
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return std::uint8_t(v * 255.0f + 0.5f);
    };
    return {c(r * m), c(g * m), c(b * m), 255};
}

// each generator returns the colour for texel (x,y) of its 16x16 tile
using Gen = Rgba (*)(int, int);

Rgba genStone(int x, int y) {
    float n = 0.86f + 0.14f * h01(x, y, 1);
    if (h01(x, y, 2) > 0.93f) n *= 0.78f; // dark specks / pits
    return shade(0.50f, 0.50f, 0.52f, n);
}

Rgba genDirt(int x, int y) {
    float n = 0.82f + 0.20f * h01(x, y, 3);
    if (h01(x, y, 4) > 0.9f) n *= 0.8f; // pebbles
    return shade(0.52f, 0.37f, 0.26f, n);
}

Rgba genGrassTop(int x, int y) {
    float n = 0.80f + 0.28f * h01(x, y, 5);
    // faint vertical blade streaks
    n *= 0.94f + 0.06f * h01(x, 0, 6);
    return shade(0.34f, 0.62f, 0.26f, n);
}

Rgba genGrassSide(int x, int y) {
    // Dirt with a grassy fringe on top (y is top-down; small y = top). The fringe is
    // deliberately deep and fades rather than ending on a hard line: seen from above,
    // terraced ground is a stack of these side faces alternating with grass tops, and a
    // mostly-brown side made every one-block step read as a stripe. A greener average
    // closes the gap between the two, and unlike a shader overlay it survives being
    // minified down to a couple of pixels out at distance.
    const int fringe = 6 + int(h01(x, 0, 8) * 3.0f);
    if (y < fringe)
        return genGrassTop(x, y);
    const int fade = 4;
    if (y < fringe + fade) {
        const Rgba g = genGrassTop(x, y);
        const Rgba d = genDirt(x, y);
        // dither between the two so the transition breaks up instead of banding
        const float t = float(y - fringe) / float(fade);
        return h01(x, y, 9) > t ? g : d;
    }
    return genDirt(x, y);
}

Rgba genWoodSide(int x, int y) {
    // vertical bark grooves
    float groove = 0.85f + 0.15f * std::sin(float(x) * 1.9f);
    groove *= 0.9f + 0.1f * h01(x, y, 10);
    return shade(0.42f, 0.29f, 0.17f, groove);
}

Rgba genWoodTop(int x, int y) {
    float dx = float(x) - 7.5f, dy = float(y) - 7.5f;
    float rings = 0.8f + 0.2f * std::sin(std::sqrt(dx * dx + dy * dy) * 2.2f);
    rings *= 0.95f + 0.05f * h01(x, y, 11);
    return shade(0.52f, 0.38f, 0.22f, rings);
}

Rgba genLeaves(int x, int y) {
    // clustered blobs of lighter/darker green
    float cluster = h01(x / 3, y / 3, 12);
    float n = 0.7f + 0.5f * cluster + 0.12f * h01(x, y, 13);
    Rgba c = shade(0.22f, 0.44f, 0.18f, n);
    // Punch gaps. Leaves are a cutout block: the holes are what you see through, what light
    // comes through, and what makes the shadow dappled instead of a solid slab. Driven off a
    // different hash than the colour so the holes do not line up with the shading, and
    // biased so the tile keeps roughly three quarters coverage — many small gaps read as
    // foliage, a few big ones read as a broken texture.
    const float hole = h01(x, y, 71) * 0.65f + h01(x / 2, y / 2, 72) * 0.35f;
    if (hole > 0.74f)
        c.a = 0;
    return c;
}

Rgba genSand(int x, int y) {
    float n = 0.9f + 0.12f * h01(x, y, 14);
    return shade(0.85f, 0.79f, 0.58f, n);
}

Rgba genGlow(int x, int y) {
    float dx = float(x) - 7.5f, dy = float(y) - 7.5f;
    float glow = 1.0f - 0.25f * std::sqrt(dx * dx + dy * dy) / 8.0f;
    glow += 0.1f * h01(x, y, 15);
    return shade(1.0f, 0.86f, 0.55f, glow);
}

Rgba genSnow(int x, int y) {
    float n = 0.92f + 0.08f * h01(x, y, 16);
    if (h01(x, y, 17) > 0.96f) n = 1.0f; // sparkle
    return shade(0.93f, 0.95f, 1.0f, n);
}

Rgba genPlanks(int x, int y) {
    // horizontal boards: darker seam every 4 rows, faint lengthwise grain
    float n = 0.86f + 0.14f * h01(x, 0, 18);
    if (y % 4 == 0) n *= 0.72f;                    // board seam
    n *= 0.94f + 0.06f * std::sin(float(x) * 1.3f); // grain
    return shade(0.60f, 0.44f, 0.27f, n);
}

Rgba genBricks(int x, int y) {
    // running-bond brick: mortar lines every 4 rows, offset the vertical joint per course
    const int row = y / 4;
    const int off = (row % 2) * 4;
    const bool mortar = (y % 4 == 0) || ((x + off) % 8 == 0);
    if (mortar)
        return shade(0.78f, 0.76f, 0.72f, 0.9f + 0.1f * h01(x, y, 19)); // pale mortar
    float n = 0.85f + 0.18f * h01(x, y, 20);
    return shade(0.62f, 0.26f, 0.20f, n); // clay red
}

// Ore is stone with a cluster of inclusions rather than a flat recolour, so it still
// reads as the rock it is embedded in. The blobs come from thresholded value noise on a
// coarse lattice, which keeps them clumped instead of salt-and-pepper.
Rgba oreSpeckle(int x, int y, int z, float r, float g, float b) {
    const Rgba base = genStone(x, y);
    const float n = h01(x / 3, y / 3, z) * 0.6f + h01(x / 2, y / 2, z + 7) * 0.4f;
    if (n < 0.62f)
        return base;
    const float d = 0.75f + 0.5f * h01(x, y, z + 21); // per-speck brightness
    return shade(r, g, b, d);
}
Rgba genCoalOre(int x, int y) { return oreSpeckle(x, y, 31, 0.13f, 0.12f, 0.14f); }
Rgba genIronOre(int x, int y) { return oreSpeckle(x, y, 47, 0.72f, 0.56f, 0.42f); }

// Lamp faces: a bright core falling off to a darker rim, so the block reads as a fitting
// with a lit centre rather than a flat colour swatch. Hue comes from the same table the
// emitted light does.
Rgba lampTexel(int hue, int x, int y) {
    const float dx = float(x) - 7.5f, dy = float(y) - 7.5f;
    const float d = std::sqrt(dx * dx + dy * dy) / 8.0f;
    const float core = 1.0f - 0.42f * d + 0.10f * h01(x, y, 70 + hue);
    const float w = 0.35f * (1.0f - d); // lift the middle toward white: that is the source
    const auto& c = blocks::kLampRGB[std::size_t(hue)];
    const float r = float(c.r) / 15.0f;
    const float g = float(c.g) / 15.0f;
    const float b = float(c.b) / 15.0f;
    return shade(r + w * (1.0f - r), g + w * (1.0f - g), b + w * (1.0f - b), core);
}

// One crack stage: branches walking out from the centre, more of them and thicker as the
// stage rises. Transparent everywhere else — this is drawn OVER the block, so anything it
// does not touch has to leave the block's own texture showing.
void fillCrack(int stage, Rgba* slot) {
    for (int i = 0; i < kTile * kTile; ++i)
        slot[i] = {0, 0, 0, 0};
    const auto mark = [&](int x, int y, std::uint8_t a) {
        if (x < 0 || x >= kTile || y < 0 || y >= kTile)
            return;
        Rgba& p = slot[y * kTile + x];
        if (a > p.a)
            p = {12, 10, 10, a};
    };
    const int branches = 2 + stage; // 2..6, not 2..10: this is a 16x16 tile
    for (int b = 0; b < branches; ++b) {
        // fixed salt per branch so a stage always contains the previous stage's cracks —
        // the animation has to look like damage accumulating, not like it is being redrawn
        float x = 7.5f, y = 7.5f;
        const float ang = (float(b) + 0.5f) * 6.2831853f / float(branches) +
                          h01(b, 0, 91) * 0.9f;
        float dx = std::cos(ang), dy = std::sin(ang);
        const int len = 3 + stage; // a branch reaching 11 texels crosses the whole face
        for (int t = 0; t < len; ++t) {
            const float wob = (h01(b, t, 92) - 0.5f) * 0.9f;
            const float na = std::atan2(dy, dx) + wob;
            dx = std::cos(na);
            dy = std::sin(na);
            x += dx;
            y += dy;
            mark(int(x), int(y), 190);
            if (stage >= 4) { // widen only on the last stage, when it is about to go
                mark(int(x) + 1, int(y), 110);
                mark(int(x), int(y) + 1, 110);
            }
        }
    }
}

// A tuft of blades rising from the bottom of the tile, transparent everywhere else. Drawn
// on two diagonal planes, so what matters is the silhouette, not the interior detail.
void fillCross(int which, Rgba* slot) {
    for (int i = 0; i < kTile * kTile; ++i)
        slot[i] = {0, 0, 0, 0};
    const int blades = which == 0 ? 7 : 5;
    for (int b = 0; b < blades; ++b) {
        const int x0 = 1 + int(h01(b, which, 80) * float(kTile - 3));
        const int h = 6 + int(h01(b, which, 81) * 8.0f);
        const float lean = (h01(b, which, 82) - 0.5f) * 0.45f;
        for (int t = 0; t < h; ++t) {
            const int x = x0 + int(lean * float(t));
            const int y = kTile - 1 - t; // y is top-down, so grow upward from the base
            if (x < 0 || x >= kTile || y < 0)
                continue;
            // darker at the root, brighter at the tip
            const float m = 0.55f + 0.45f * float(t) / float(h);
            slot[y * kTile + x] = shade(0.30f, 0.58f, 0.22f, m);
            slot[y * kTile + x].a = 255;
        }
        if (which == 1 && b < 3) { // flower heads on the first few blades
            const int y = kTile - 1 - h;
            for (int dy = -1; dy <= 0; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = x0 + int(lean * float(h)) + dx;
                    const int yy = y + dy;
                    if (x < 0 || x >= kTile || yy < 0 || yy >= kTile)
                        continue;
                    slot[yy * kTile + x] = shade(0.92f, 0.82f, 0.30f, 1.0f);
                    slot[yy * kTile + x].a = 255;
                }
        }
    }
}

// Loose pebbles: a few grey tones scattered per texel rather than one tinted noise field,
// so it reads as aggregate next to stone instead of as dirty stone.
Rgba genGravel(int x, int y) {
    const float pick = h01(x, y, 61);
    const float tone = pick < 0.30f ? 0.42f : (pick < 0.62f ? 0.55f : (pick < 0.86f ? 0.66f : 0.34f));
    const float n = 0.90f + 0.16f * h01(x, y, 62);
    return shade(tone, tone * 0.98f, tone * 0.94f, n);
}

constexpr std::array<Gen, kNamedLayers> kGens = {genStone,    genDirt,    genGrassTop,
                                            genGrassSide, genGlow,   genWoodSide,
                                            genWoodTop,  genLeaves,  genSand,
                                            genSnow,    genPlanks,  genBricks,
                                            genCoalOre, genIronOre, genGravel};

} // namespace

// Loads <dir>/<name>.png into the 16x16 layer slot, nearest-resampling any size down to
// the tile. Returns false if the file is missing/unreadable (caller uses procedural).
bool loadLayerPng(const std::filesystem::path& path, Rgba* out) {
    int w = 0, h = 0, comp = 0;
    stbi_uc* data = stbi_load(path.string().c_str(), &w, &h, &comp, 4);
    if (!data)
        return false;
    for (int y = 0; y < kTile; ++y)
        for (int x = 0; x < kTile; ++x) {
            const int sx = x * w / kTile;
            const int sy = y * h / kTile;
            const stbi_uc* p = data + (std::size_t(sy) * w + sx) * 4;
            out[y * kTile + x] = {p[0], p[1], p[2], p[3]};
        }
    stbi_image_free(data);
    return true;
}

unsigned int createBlockTextureArray(const std::filesystem::path& textureDir) {
    std::array<Rgba, kTile * kTile * kLayers> pixels{};
    int loaded = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        Rgba* slot = pixels.data() + std::size_t(layer) * kTile * kTile;
        if (layer >= kCrossLayer0 && layer < kCrackLayer0) { // foliage: generated, has alpha
            fillCross(layer - kCrossLayer0, slot);
            continue;
        }
        if (layer >= kCrackLayer0) { // crack stages: generated, never loaded
            fillCrack(layer - kCrackLayer0, slot);
            continue;
        }
        if (layer >= kLampLayer0) { // lamps are generated only — one per hue, no filenames
            for (int y = 0; y < kTile; ++y)
                for (int x = 0; x < kTile; ++x)
                    slot[y * kTile + x] = lampTexel(layer - kLampLayer0, x, y);
            continue;
        }
        const std::filesystem::path png = textureDir / (std::string(kNames[layer]) + ".png");
        if (loadLayerPng(png, slot)) {
            ++loaded;
        } else {
            for (int y = 0; y < kTile; ++y)
                for (int x = 0; x < kTile; ++x)
                    slot[y * kTile + x] = kGens[layer](x, y);
        }
    }
    if (loaded > 0)
        VX_INFO("block textures: loaded %d/%d from %s (rest procedural)", loaded, kLayers,
                textureDir.string().c_str());

    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &tex);
    const int mips = 5; // 16 -> 1
    glTextureStorage3D(tex, mips, GL_RGBA8, kTile, kTile, kLayers);
    glTextureSubImage3D(tex, 0, 0, 0, 0, kTile, kTile, kLayers, GL_RGBA, GL_UNSIGNED_BYTE,
                        pixels.data());
    glGenerateTextureMipmap(tex);

    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // crisp pixels up close
    GLfloat aniso = 0.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &aniso);
    glTextureParameterf(tex, GL_TEXTURE_MAX_ANISOTROPY, aniso);
    return tex;
}

int blockTopLayer(int b) {
    switch (b) {
        case blocks::Stone: return 0;
        case blocks::Dirt: return 1;
        case blocks::Grass: return 2;
        case blocks::Glow: return 4;
        case blocks::Wood: return 6;
        case blocks::Leaves: return 7;
        case blocks::Sand: return 8;
        case blocks::Snow: return 9;
        case blocks::Planks: return 10;
        case blocks::Bricks: return 11;
        case blocks::CoalOre: return 12;
        case blocks::IronOre: return 13;
        case blocks::Gravel: return 14;
        default: break;
    }
    if (blocks::isLamp(BlockId(b)))
        return kLampLayer0 + (b - blocks::LampFirst);
    if (b == blocks::GrassTuft)
        return kCrossLayer0;
    if (b == blocks::Flower)
        return kCrossLayer0 + 1;
    return 0;
}

int crackStages() { return kCrackStages; }
int crackLayer(int stage) {
    const int s = stage < 0 ? 0 : (stage >= kCrackStages ? kCrackStages - 1 : stage);
    return kCrackLayer0 + s;
}

} // namespace vx
