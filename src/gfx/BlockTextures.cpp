#include "gfx/BlockTextures.hpp"

#include "core/Log.hpp"

#include <glad/gl.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <array>
#include <cmath>
#include <cstdint>

namespace vx {

namespace {

constexpr int kTile = 16;
constexpr int kLayers = 14;

// filename (without extension) per layer, looked up in assets/textures/
constexpr const char* kNames[kLayers] = {"stone",    "dirt",     "grass_top", "grass_side",
                                         "glow",     "wood_side", "wood_top", "leaves",
                                         "sand",     "snow",     "planks",    "bricks",
                                         "coal_ore", "iron_ore"};

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
    return shade(0.22f, 0.44f, 0.18f, n);
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

constexpr std::array<Gen, kLayers> kGens = {genStone,    genDirt,    genGrassTop,
                                            genGrassSide, genGlow,   genWoodSide,
                                            genWoodTop,  genLeaves,  genSand,
                                            genSnow,    genPlanks,  genBricks,
                                            genCoalOre, genIronOre};

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

} // namespace vx
