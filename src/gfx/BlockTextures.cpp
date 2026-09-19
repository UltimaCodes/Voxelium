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
// Coast, desert and reef tiles. Appended after the crack stages so every layer number the
// shaders already hard-code (leaves 7, lamps 15..39, cross 40/41) stays where it was.
constexpr int kBiomeLayer0 = kCrackLayer0 + kCrackStages;
constexpr int kBiomeLayers = 28;
constexpr int kLayers = kBiomeLayer0 + kBiomeLayers;

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
    float n = 0.76f + 0.5f * cluster + 0.12f * h01(x, y, 13);
    // Lit tops read as near-black next to the grass at 0.22/0.44/0.18. Lifted a little so a
    // sunlit canopy is visibly lit; the darker clusters still carry the depth.
    Rgba c = shade(0.25f, 0.49f, 0.19f, n);
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


// ---- coast / desert / reef ----------------------------------------------------------------

using Fill = void (*)(Rgba*);

template <Gen G>
void fillWith(Rgba* slot) {
    for (int y = 0; y < kTile; ++y)
        for (int x = 0; x < kTile; ++x)
            slot[y * kTile + x] = G(x, y);
}

void clearTile(Rgba* slot) {
    for (int i = 0; i < kTile * kTile; ++i)
        slot[i] = {0, 0, 0, 0};
}

void put(Rgba* slot, int x, int y, Rgba c) {
    if (x >= 0 && x < kTile && y >= 0 && y < kTile)
        slot[y * kTile + x] = c;
}

Rgba solid(float r, float g, float b, float m = 1.0f) {
    Rgba c = shade(r, g, b, m);
    c.a = 255;
    return c;
}

// Beach sand: fine warm grain with faint wind ripples and the odd stray fleck. It used to be
// a tile of scallops, conches and clams, which read as a pattern stamped over the beach; real
// sand is mostly just sand, and the shells that do sit on it are the Shell blocks. Repetition
// across the beach is broken up in the shader (per-block rotation and tint), not here.
Rgba genBeachSand(int x, int y) {
    float n = 0.93f + 0.10f * h01(x, y, 101);
    // wind ripples: slow diagonal waves, a few percent of brightness
    n += 0.035f * std::sin((float(x) + float(y) * 0.6f) * 0.85f + h01(y / 5, 3, 103) * 3.0f);
    const float g = h01(x, y, 102);
    if (g > 0.955f)
        n *= 0.84f; // a darker grain
    else if (g < 0.03f)
        n *= 1.10f; // a paler one
    // Warm cream. Kept below the white a sunlit beach blows out to: 0.88/0.80/0.62 was a white
    // sheet, 0.64/0.53/0.34 was right for luminance but greyish once lit.
    Rgba c = shade(0.70f, 0.55f, 0.31f, n);
    // very rare crushed-shell fleck, a pixel or two
    const float f = h01(x, y, 104);
    if (f > 0.992f)
        c = shade(0.93f, 0.88f, 0.80f, 1.0f);
    else if (f < 0.004f)
        c = shade(0.86f, 0.66f, 0.62f, 1.0f);
    return c;
}

Rgba genSandstoneSide(int x, int y) {
    // strata: bands of slightly different tone, each with its own grain
    const float band = std::sin(float(y) * 0.9f + h01(0, y / 3, 111) * 3.0f);
    float n = 0.90f + 0.06f * band + 0.07f * h01(x, y, 112);
    if (y % 5 == 0)
        n *= 0.88f; // a bedding line
    return shade(0.86f, 0.74f, 0.50f, n);
}

Rgba genSandstoneTop(int x, int y) {
    const float n = 0.95f + 0.07f * h01(x, y, 113);
    return shade(0.88f, 0.77f, 0.53f, n);
}

Rgba genCactusSide(int x, int y) {
    // vertical ribs: a dark groove every 4 columns, spine dots on the ridges
    float n = 0.85f + 0.10f * h01(x, y, 121);
    const int col = x & 3;
    if (col == 0)
        n *= 0.72f;
    Rgba c = shade(0.22f, 0.50f, 0.22f, n);
    if (col == 2 && (y % 4) == 1)
        c = shade(0.93f, 0.92f, 0.78f, 1.0f); // a spine
    return c;
}

Rgba genCactusTop(int x, int y) {
    const float dx = float(x) - 7.5f, dy = float(y) - 7.5f;
    const float r = std::sqrt(dx * dx + dy * dy);
    float n = 0.85f + 0.10f * h01(x, y, 122);
    if (r > 6.2f)
        n *= 0.7f; // rind
    Rgba c = shade(0.24f, 0.54f, 0.24f, n);
    if (r < 1.6f)
        c = shade(0.90f, 0.88f, 0.70f, 1.0f); // the crown of spines
    return c;
}

void fillDeadBush(Rgba* slot) {
    clearTile(slot);
    const Rgba twig = solid(0.45f, 0.32f, 0.18f);
    const Rgba dry = solid(0.55f, 0.42f, 0.24f);
    for (int t = 0; t < 8; ++t)
        put(slot, 7, 15 - t, twig); // the trunk
    const auto branch = [&](int x0, int y0, int dx, int len, Rgba c) {
        int x = x0, y = y0;
        for (int i = 0; i < len; ++i) {
            put(slot, x, y, c);
            x += dx;
            if (i % 2 == 1)
                --y;
        }
    };
    branch(7, 9, -1, 6, dry);
    branch(7, 9, 1, 6, twig);
    branch(7, 11, -1, 5, twig);
    branch(7, 11, 1, 4, dry);
    branch(7, 7, -1, 3, twig);
    branch(7, 7, 1, 4, dry);
}

Rgba genPalmSide(int x, int y) {
    // Palm bark is grey-brown and crosshatched with leaf scars: short dark dashes in staggered
    // rows over vertical fibre. Two earlier versions (single scar rows, then wide light bands)
    // both read as stacked planks rather than a trunk.
    float m = 0.86f + 0.10f * h01(x, 0, 131) + 0.08f * h01(x, y, 132);
    Rgba c = shade(0.56f, 0.45f, 0.31f, m);
    const int row = y % 5;
    const bool dash = row == 0 && (((x + (y / 5) * 3) / 3) & 1) == 0;
    if (dash)
        c = shade(0.30f, 0.23f, 0.15f, 0.9f + 0.1f * h01(x, y, 133));
    else if (row == 1 && (((x + (y / 5) * 3) / 3) & 1) == 0)
        c = shade(0.44f, 0.35f, 0.24f, 1.0f); // the shadow under each scar
    if (x == 0 || x == 15)
        c = shade(0.40f, 0.31f, 0.21f, 1.0f); // a darker edge so the corners of the trunk read
    return c;
}

Rgba genPalmTop(int x, int y) {
    const float dx = float(x) - 7.5f, dy = float(y) - 7.5f;
    const float rings = 0.8f + 0.2f * std::sin(std::sqrt(dx * dx + dy * dy) * 2.6f);
    return shade(0.62f, 0.52f, 0.34f, rings * (0.95f + 0.05f * h01(x, y, 133)));
}

Rgba genPalmLeaves(int x, int y) {
    // a frond seen flat: a central rib with leaflets angling off it and gaps between them, so
    // the cutout reads as many narrow leaves rather than one green sheet
    const float dx = std::abs(float(x) - 7.5f);
    const bool rib = dx < 0.9f;
    const int row = int(std::floor(float(y) + dx * 0.55f)); // leaflets sweep back toward the tip
    const bool leaflet = (row % 3) != 2 && dx < 7.5f;
    Rgba c = shade(0.20f, 0.56f, 0.20f, 0.78f + 0.4f * h01(x / 2, y / 2, 141));
    if (rib)
        c = shade(0.34f, 0.66f, 0.28f, 1.0f);
    c.a = (rib || (leaflet && h01(x, y, 142) > 0.16f)) ? 255 : 0;
    return c;
}

Rgba genCoconut(int x, int y) {
    const float dx = float(x) - 7.5f, dy = float(y) - 7.5f;
    const float n = 0.80f + 0.30f * h01(x, y, 151);
    Rgba c = shade(0.42f, 0.28f, 0.16f, n);
    if (std::sqrt(dx * dx + dy * dy) > 7.0f)
        c = shade(0.30f, 0.20f, 0.11f, 1.0f);
    if ((x == 5 && y == 5) || (x == 9 && y == 5) || (x == 7 && y == 8))
        c = shade(0.12f, 0.08f, 0.05f, 1.0f); // the three eyes
    return c;
}

// seagrass: a clump of long blades that lean and cross
void fillSeagrass(Rgba* slot) {
    clearTile(slot);
    for (int b = 0; b < 7; ++b) {
        const int x0 = 1 + int(h01(b, 0, 161) * 13.0f);
        const int h = 8 + int(h01(b, 0, 162) * 7.0f);
        const float lean = (h01(b, 0, 163) - 0.5f) * 0.6f;
        for (int t = 0; t < h; ++t) {
            const int x = x0 + int(lean * float(t) + std::sin(float(t) * 0.7f + float(b)) * 0.8f);
            const int y = kTile - 1 - t;
            const float m = 0.65f + 0.35f * float(t) / float(h);
            put(slot, x, y, solid(0.16f, 0.58f, 0.36f, m));
            if (t > h / 2)
                put(slot, x + 1, y, solid(0.20f, 0.66f, 0.40f, m)); // the blade widens
        }
    }
}

// kelp: two long strands that start and end on the same column, so blocks stacked on top of
// each other read as one continuous stalk
void fillKelp(Rgba* slot) {
    clearTile(slot);
    for (int strand = 0; strand < 2; ++strand) {
        const float base = strand == 0 ? 5.0f : 10.5f;
        for (int y = 0; y < kTile; ++y) {
            const float w = base + std::sin((float(y) / float(kTile)) * 6.2831853f +
                                            float(strand) * 2.1f) * 1.6f;
            const int x = int(std::floor(w));
            put(slot, x, y, solid(0.40f, 0.44f, 0.10f, 0.85f + 0.2f * h01(x, y, 171)));
            put(slot, x + 1, y, solid(0.34f, 0.40f, 0.09f, 0.9f));
            if ((y % 4) == 1) { // a leaf off the side, alternating
                const int dir = ((y / 4) & 1) ? 1 : -1;
                put(slot, x + dir * 2, y, solid(0.52f, 0.56f, 0.14f, 1.0f));
                put(slot, x + dir * 3, y - 1, solid(0.52f, 0.56f, 0.14f, 1.0f));
                put(slot, x + dir * 2, y + 1, solid(0.44f, 0.48f, 0.12f, 1.0f));
            }
        }
    }
}

Rgba genBrainCoral(int x, int y) {
    // the meandering ridges of a brain: two crossed sine fields
    const float u = float(x), v = float(y);
    const float r = std::sin(u * 1.25f + std::sin(v * 0.95f) * 1.9f) +
                    std::sin(v * 1.20f + std::sin(u * 1.10f) * 1.9f);
    const float m = r > 0.55f ? 1.12f : (r < -0.7f ? 0.72f : 0.94f);
    return shade(0.92f, 0.46f, 0.62f, m * (0.94f + 0.08f * h01(x, y, 181)));
}

Rgba genTubeCoral(int x, int y) {
    // a grid of open tubes: a pale rim round a dark mouth
    const int cx = (x & 7) - 4, cy = (y & 7) - 4;
    const float d = std::sqrt(float(cx * cx + cy * cy));
    if (d < 1.4f)
        return shade(0.05f, 0.12f, 0.32f, 1.0f);
    if (d < 2.6f)
        return shade(0.50f, 0.72f, 1.0f, 1.0f);
    return shade(0.18f, 0.44f, 0.88f, 0.86f + 0.16f * h01(x, y, 191));
}

Rgba genFireCoral(int x, int y) {
    // vertical branching, hot at the tips
    const float n = 0.86f + 0.22f * h01(x / 2, y, 201);
    const bool streak = (h01(x, 0, 202) > 0.5f) ^ ((y / 3) % 2 == 0);
    Rgba c = shade(0.96f, streak ? 0.42f : 0.28f, 0.12f, n);
    if (h01(x, y, 203) > 0.9f)
        c = shade(1.0f, 0.78f, 0.25f, 1.0f);
    return c;
}

Rgba genHornCoral(int x, int y) {
    // a tree of dark forking lines over yellow
    const bool line = ((x * 3 + y * 2) % 7 == 0) || ((x * 2 - y * 3 + 64) % 9 == 0);
    if (line)
        return shade(0.62f, 0.44f, 0.08f, 1.0f);
    return shade(0.96f, 0.80f, 0.24f, 0.88f + 0.16f * h01(x, y, 211));
}

Rgba genBubbleCoral(int x, int y) {
    // clusters of round bubbles with a bright glint each
    const float bx = float(((x + 3) / 5) * 5) - 0.5f + (h01(x / 5, y / 5, 221) - 0.5f) * 2.0f;
    const float by = float(((y + 3) / 5) * 5) - 0.5f + (h01(y / 5, x / 5, 222) - 0.5f) * 2.0f;
    const float d = std::sqrt((float(x) - bx) * (float(x) - bx) + (float(y) - by) * (float(y) - by));
    if (d < 1.0f)
        return shade(0.92f, 0.78f, 1.0f, 1.0f);
    if (d < 2.7f)
        return shade(0.72f, 0.42f, 0.90f, 1.0f);
    return shade(0.40f, 0.20f, 0.60f, 0.9f);
}

// A coral fan: veins radiate from the foot of the tile, joined by a lace of webbing.
// Transparent outside the fan so it draws as a frond, not a block.
void fillFan(Rgba* slot, float r, float g, float b) {
    clearTile(slot);
    for (int y = 0; y < kTile; ++y)
        for (int x = 0; x < kTile; ++x) {
            const float dx = float(x) - 7.5f, dy = 15.5f - float(y);
            const float d = std::sqrt(dx * dx + dy * dy);
            if (d > 14.0f || d < 1.5f || dy < 0.5f)
                continue;
            const float ang = std::atan2(dx, dy);
            if (std::abs(ang) > 1.05f)
                continue;
            const float vein = std::abs(std::sin(ang * 9.0f));
            const bool web = std::fmod(d, 3.4f) < 0.9f;
            if (vein > 0.82f || web || d < 3.0f)
                put(slot, x, y, solid(r, g, b, 0.75f + 0.35f * (d / 14.0f)));
        }
}
void fillFanPink(Rgba* slot) { fillFan(slot, 0.96f, 0.42f, 0.66f); }
void fillFanBlue(Rgba* slot) { fillFan(slot, 0.30f, 0.55f, 0.98f); }
void fillFanOrange(Rgba* slot) { fillFan(slot, 0.98f, 0.55f, 0.16f); }

// a conch standing on the sand, and nothing else on the tile
void fillShell(Rgba* slot) {
    clearTile(slot);
    const int cx = 7, cy = 13;
    for (int y = -5; y <= 2; ++y)
        for (int x = -4; x <= 4; ++x) {
            const float rx = float(x) / 4.2f, ry = float(y) / 4.2f;
            if (rx * rx + ry * ry * 1.4f > 1.0f)
                continue;
            const bool ridge = ((x + y * 2 + 16) & 3) == 0;
            put(slot, cx + x, cy + y,
                ridge ? solid(0.78f, 0.52f, 0.44f, 1.0f) : solid(0.98f, 0.86f, 0.80f, 1.0f));
        }
}

Rgba genGlowCoral(int x, int y) {
    // organic cells with bright cores: the coral is meant to read as a light source
    const float cx = float(((x + 2) / 6) * 6) - 0.5f + (h01(x / 6, y / 6, 231) - 0.5f) * 2.0f;
    const float cy = float(((y + 2) / 6) * 6) - 0.5f + (h01(y / 6, x / 6, 232) - 0.5f) * 2.0f;
    const float d = std::sqrt((float(x) - cx) * (float(x) - cx) + (float(y) - cy) * (float(y) - cy));
    float glow = 1.0f - 0.16f * d + 0.08f * h01(x, y, 233);
    if (glow < 0.4f)
        glow = 0.4f;
    return shade(0.32f, 1.0f, 0.95f, glow);
}

Rgba genSpruceSide(int x, int y) {
    // dark, deeply furrowed bark
    float n = 0.78f + 0.22f * std::sin(float(x) * 2.3f + h01(0, y / 4, 161) * 2.0f);
    n *= 0.88f + 0.12f * h01(x, y, 162);
    if (h01(x, y, 163) > 0.93f)
        n *= 0.7f;
    return shade(0.30f, 0.21f, 0.14f, n);
}

Rgba genSpruceTop(int x, int y) {
    const float dx = float(x) - 7.5f, dy = float(y) - 7.5f;
    const float r = std::sqrt(dx * dx + dy * dy);
    float n = 0.80f + 0.20f * std::sin(r * 2.4f);
    if (r > 6.4f)
        n *= 0.66f; // bark rim
    return shade(0.50f, 0.37f, 0.22f, n * (0.95f + 0.05f * h01(x, y, 164)));
}

Rgba genBirchSide(int x, int y) {
    // pale papery bark with short black scars, each a dash or two wide
    float n = 0.93f + 0.07f * h01(x, y, 171);
    Rgba c = shade(0.88f, 0.87f, 0.82f, n);
    const int band = y / 3;
    const float pick = h01(x / 2, band, 172);
    if (pick > 0.66f && (y % 3) != 1)
        c = shade(0.10f, 0.10f, 0.09f, 0.8f + 0.4f * h01(x, y, 173));
    else if (pick > 0.55f && (y % 3) == 1)
        c = shade(0.55f, 0.53f, 0.48f, 1.0f);
    return c;
}

Rgba genBirchTop(int x, int y) {
    const float dx = float(x) - 7.5f, dy = float(y) - 7.5f;
    const float r = std::sqrt(dx * dx + dy * dy);
    float n = 0.84f + 0.16f * std::sin(r * 2.6f);
    Rgba c = shade(0.76f, 0.66f, 0.44f, n);
    if (r > 6.4f)
        c = shade(0.86f, 0.85f, 0.80f, 1.0f);
    return c;
}

Rgba genSpruceLeaves(int x, int y) {
    // needles: dark blue-green, with tighter gaps than oak
    const float stroke = h01((x + y) / 2, y / 2, 181);
    Rgba c = shade(0.17f, 0.35f, 0.15f, 0.72f + 0.55f * stroke + 0.10f * h01(x, y, 182));
    const float hole = h01(x, y, 183) * 0.7f + h01(x / 2, y / 2, 184) * 0.3f;
    if (hole > 0.78f)
        c.a = 0;
    return c;
}

Rgba genBirchLeaves(int x, int y) {
    // small bright leaves: light yellow-green in tight clusters
    const float cluster = h01(x / 2, y / 2, 191);
    Rgba c = shade(0.44f, 0.60f, 0.22f, 0.78f + 0.45f * cluster + 0.10f * h01(x, y, 192));
    const float hole = h01(x, y, 193) * 0.65f + h01(x / 2, y / 2, 194) * 0.35f;
    if (hole > 0.72f)
        c.a = 0;
    return c;
}

constexpr std::array<Fill, kBiomeLayers> kBiomeFill = {
    fillWith<genBeachSand>,     // 47 beach sand
    fillWith<genSandstoneSide>, // 48
    fillWith<genSandstoneTop>,  // 49
    fillWith<genCactusSide>,    // 50
    fillWith<genCactusTop>,     // 51
    fillDeadBush,               // 52
    fillWith<genPalmSide>,      // 53
    fillWith<genPalmTop>,       // 54
    fillWith<genPalmLeaves>,    // 55
    fillWith<genCoconut>,       // 56
    fillSeagrass,               // 57
    fillKelp,                   // 58
    fillWith<genBrainCoral>,    // 59
    fillWith<genTubeCoral>,     // 60
    fillWith<genFireCoral>,     // 61
    fillWith<genHornCoral>,     // 62
    fillWith<genBubbleCoral>,   // 63
    fillFanPink,                // 64
    fillFanBlue,                // 65
    fillFanOrange,              // 66
    fillShell,                  // 67
    fillWith<genGlowCoral>,     // 68
    fillWith<genSpruceSide>,    // 69
    fillWith<genSpruceTop>,     // 70
    fillWith<genBirchSide>,     // 71
    fillWith<genBirchTop>,      // 72
    fillWith<genSpruceLeaves>,  // 73
    fillWith<genBirchLeaves>,   // 74
};

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
        if (layer >= kBiomeLayer0) { // coast, desert, reef: generated only
            kBiomeFill[std::size_t(layer - kBiomeLayer0)](slot);
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
            // The bundled dirt and bark are rosy browns; under a blue sky ambient they read
            // as mauve and lavender. Pushed toward a yellower brown (red kept, green up, blue
            // down) so a lit dirt bank looks like earth.
            // The bundled sand is speckled pink and brown at full contrast, which reads as noise
            // across a whole desert. Pulled 35% toward its own average.
            if (layer == 8) {
                float mr = 0, mg = 0, mb = 0;
                for (int i = 0; i < kTile * kTile; ++i) {
                    mr += slot[i].r;
                    mg += slot[i].g;
                    mb += slot[i].b;
                }
                mr /= float(kTile * kTile);
                mg /= float(kTile * kTile);
                mb /= float(kTile * kTile);
                for (int i = 0; i < kTile * kTile; ++i) {
                    slot[i].r = std::uint8_t(mr + (float(slot[i].r) - mr) * 0.65f);
                    slot[i].g = std::uint8_t(mg + (float(slot[i].g) - mg) * 0.65f);
                    slot[i].b = std::uint8_t(mb + (float(slot[i].b) - mb) * 0.65f);
                }
            }
            if (layer == 1 || layer == 3 || layer == 5 || layer == 6)
                for (int i = 0; i < kTile * kTile; ++i) {
                    Rgba& t = slot[i];
                    const auto clampU = [](float v) {
                        return std::uint8_t(v < 0 ? 0 : (v > 255 ? 255 : v));
                    };
                    // grass_side (3) carries green in its top: only the brown part is warmed
                    if (t.g > t.r + 10 && t.g > t.b + 10)
                        continue;
                    t.r = clampU(float(t.r) * 1.03f);
                    t.g = clampU(float(t.g) * 1.13f);
                    t.b = clampU(float(t.b) * 0.76f);
                }
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
    switch (b) {
        case blocks::BeachSand: return kBiomeLayer0;
        case blocks::Sandstone: return kBiomeLayer0 + 2;
        case blocks::Cactus: return kBiomeLayer0 + 4;
        case blocks::DeadBush: return kBiomeLayer0 + 5;
        case blocks::PalmWood: return kBiomeLayer0 + 7;
        case blocks::PalmLeaves: return kBiomeLayer0 + 8;
        case blocks::Coconut: return kBiomeLayer0 + 9;
        case blocks::SpruceWood: return kBiomeLayer0 + 23;
        case blocks::BirchWood: return kBiomeLayer0 + 25;
        case blocks::SpruceLeaves: return kBiomeLayer0 + 26;
        case blocks::BirchLeaves: return kBiomeLayer0 + 27;
        default: break;
    }
    if (b >= blocks::Seagrass && b <= blocks::GlowCoral)
        return kBiomeLayer0 + 10 + (b - blocks::Seagrass);
    return 0;
}

int crackStages() { return kCrackStages; }
int crackLayer(int stage) {
    const int s = stage < 0 ? 0 : (stage >= kCrackStages ? kCrackStages - 1 : stage);
    return kCrackLayer0 + s;
}

} // namespace vx
