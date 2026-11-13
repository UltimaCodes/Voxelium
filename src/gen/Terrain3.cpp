#include "gen/Terrain3.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <climits>
#include <cmath>

namespace vx {

namespace {

constexpr int kSeaLevel = 62;
constexpr float kCoastField = -0.06f;
// The coast further off, as version 2 finds it: the field about 480 blocks out in eight directions.
constexpr int kLookDirs = 8;
constexpr int kLookX[kLookDirs] = {480, -480, 0, 0, 340, 340, -340, -340};
constexpr int kLookZ[kLookDirs] = {0, 0, 480, -480, 340, -340, 340, -340};

constexpr float kPlateCell = 4200.0f;
constexpr float kNarrow = 420.0f; // softness of a plate edge: the massif
constexpr float kWide = 1100.0f;  // and its foothills

constexpr float kWindX = 0.83f, kWindZ = 0.55f;

float smooth01(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

std::uint32_t hash2(std::int32_t x, std::int32_t z, std::uint32_t salt) {
    std::uint32_t h = std::uint32_t(x) * 0x8DA6B343u ^ std::uint32_t(z) * 0xD8163841u ^ salt * 0xCB1AB31Fu;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

float hash01(std::uint32_t h) { return float(h & 0xFFFFu) / 65535.0f; }

// Catmull-Rom through four values.
float cubic(float a, float b, float c, float d, float t) {
    return b + 0.5f * t * (c - a + t * (2.0f * a - 5.0f * b + 4.0f * c - d + t * (3.0f * (b - c) + d - a)));
}

// A grid field read at (fx, fz) in lattice cells, where cell 0 starts at the grid's second point.
float cubic2(const float* v, int n, float fx, float fz) {
    const int ix = std::clamp(int(std::floor(fx)), 0, n - 4), iz = std::clamp(int(std::floor(fz)), 0, n - 4);
    const float tx = fx - float(ix), tz = fz - float(iz);
    float row[4];
    for (int j = 0; j < 4; ++j) {
        const float* r = v + std::size_t(iz + j) * n + ix;
        row[j] = cubic(r[0], r[1], r[2], r[3], tx);
    }
    return cubic(row[0], row[1], row[2], row[3], tz);
}

float floorDivF(float a, float b) { return std::floor(a / b); }

float valueNoise(float x, float z, std::uint32_t salt) {
    const float fx0 = std::floor(x), fz0 = std::floor(z);
    const std::int32_t ix = std::int32_t(fx0), iz = std::int32_t(fz0);
    const float fx = x - fx0, fz = z - fz0;
    const float sx = fx * fx * (3.0f - 2.0f * fx), sz = fz * fz * (3.0f - 2.0f * fz);
    const float a = hash01(hash2(ix, iz, salt)), b = hash01(hash2(ix + 1, iz, salt));
    const float c = hash01(hash2(ix, iz + 1, salt)), d = hash01(hash2(ix + 1, iz + 1, salt));
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sz;
}

// A monotone cubic through a table: smooth through every point and never overshooting.
float curve(const float* xs, const float* ys, int count, float x) {
    if (x <= xs[0])
        return ys[0];
    if (x >= xs[count - 1])
        return ys[count - 1];
    int i = 1;
    while (x > xs[i])
        ++i;
    const auto slope = [&](int k) { return (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]); };
    const auto tangent = [&](int k) {
        if (k == 0)
            return slope(0);
        if (k == count - 1)
            return slope(count - 2);
        const float a = slope(k - 1), b = slope(k);
        return a * b <= 0.0f ? 0.0f : 2.0f / (1.0f / a + 1.0f / b);
    };
    const float span = xs[i] - xs[i - 1];
    const float t = (x - xs[i - 1]) / span;
    const float t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i - 1] + (t3 - 2 * t2 + t) * span * tangent(i - 1) +
           (-2 * t3 + 3 * t2) * ys[i] + (t3 - t2) * span * tangent(i);
}

float smoothMax(float a, float b, float k) {
    const float hk = std::max(k - std::abs(a - b), 0.0f) / k;
    return std::max(a, b) + hk * hk * k * 0.25f;
}

// Sheet of parallel gullies at one scale, after Rune Skovbo Johansen's erosion filter and the
// earlier work it builds on: scattered wave sources, each laying stripes that run straight down the
// local slope, blended so the stripes wander and fork. No simulation and no neighbour reads.
// (dx, dz) is the direction across the slope; p is in cells. Value in about -1..1, gradient per cell.
struct Gully {
    float v, gx, gz;
};
Gully gullies(float px, float pz, float dx, float dz, std::uint32_t salt) {
    const float ix = std::floor(px + 0.5f), iz = std::floor(pz + 0.5f);
    const float fx = px - ix, fz = pz - iz;
    const std::int32_t cx = std::int32_t(ix), cz = std::int32_t(iz);
    float v = 0.0f, gx = 0.0f, gz = 0.0f, wt = 0.0f;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            const std::uint32_t h = hash2(cx + i, cz + j, salt);
            const float ox = fx - float(i) - (float(h & 255u) / 255.0f - 0.5f) * 0.8f;
            const float oz = fz - float(j) - (float((h >> 8) & 255u) / 255.0f - 0.5f) * 0.8f;
            const float d = ox * ox + oz * oz;
            if (d >= 2.25f)
                continue;
            float w = 1.0f - d / 2.25f;
            w = w * w * w;
            wt += w;
            // one period of cos and sin from parabolas, cheap and smooth enough for a height
            const float mag = ox * dx + oz * dz;
            const float t = mag - std::floor(mag + 0.5f);
            const float at = std::abs(t);
            const float c = at < 0.25f ? 1.0f - 16.0f * at * at : 16.0f * (0.5f - at) * (0.5f - at) - 1.0f;
            float u = t - 0.25f;
            u -= std::floor(u + 0.5f);
            const float au = std::abs(u);
            const float s = au < 0.25f ? 1.0f - 16.0f * au * au : 16.0f * (0.5f - au) * (0.5f - au) - 1.0f;
            v += c * w;
            gx -= s * dx * w;
            gz -= s * dz * w;
        }
    if (wt > 0.0f) {
        v /= wt;
        gx /= wt;
        gz /= wt;
    }
    return {v, gx, gz};
}

// Layers of gullies on a slope, each following the slope the ones before it left, so small ones
// branch off big ones. Returns the change in height.
float erode(float x, float z, float slopeX, float slopeZ, float amp, int step, std::uint32_t seed) {
    constexpr int kOctaves = 4; // a fifth cut thin blades into the crests
    float a = amp;
    float wavelength = 210.0f;
    float h = 0.0f, hx = 0.0f, hz = 0.0f;
    for (int o = 0; o < kOctaves; ++o) {
        if (wavelength < float(step) * 2.5f)
            break;
        const float f = 1.0f / wavelength;
        const float sx = slopeX + hx, sz = slopeZ + hz;
        const float len = std::sqrt(sx * sx + sz * sz);
        if (len < 1e-4f)
            break;
        const Gully g = gullies(x * f, z * f, -sz / len, sx / len, seed + std::uint32_t(o) * 31u);
        h += a * g.v;
        // gradient in blocks: stripes bend the slope the next octave sees
        hx += a * g.gx * f * 6.28f * 0.5f;
        hz += a * g.gz * f * 6.28f * 0.5f;
        a *= 0.5f;
        wavelength *= 0.47f;
    }
    return h;
}

auto fbm(float scale, int octaves, float gain) {
    auto base = FastNoise::New<FastNoise::Simplex>();
    base->SetScale(scale);
    auto f = FastNoise::New<FastNoise::FractalFBm>();
    f->SetSource(base);
    f->SetOctaveCount(octaves);
    f->SetGain(gain);
    return f;
}

constexpr std::size_t kTileSlots = 512;

std::int64_t tileKey(int tx, int tz) { return (std::int64_t(tx) << 32) ^ std::int64_t(std::uint32_t(tz)); }
std::size_t tileBucket(int tx, int tz) { return (std::size_t(std::uint32_t(tx)) * 73u + std::uint32_t(tz) * 179u) % kTileSlots; }

} // namespace

Terrain3::Terrain3(int seed) : m_seed(seed) {
    m_plateTiles.slots.resize(kTileSlots);
    m_climateTiles.slots.resize(kTileSlots);
    m_riverTiles.slots.resize(kTileSlots);
    {
        // Continents thousands of blocks across, with fine enough octaves for islands off them.
        auto warp = FastNoise::New<FastNoise::DomainWarpGradient>();
        warp->SetSource(fbm(6400.0f, 5, 0.48f));
        warp->SetScale(1900.0f);
        warp->SetWarpAmplitude(640.0f);
        m_cont = warp;
    }
    m_warpX = fbm(2600.0f, 2, 0.35f);
    m_warpZ = fbm(2600.0f, 2, 0.35f);
    {
        auto h = FastNoise::New<FastNoise::Simplex>();
        h->SetScale(215.0f);
        auto f = FastNoise::New<FastNoise::FractalFBm>();
        f->SetSource(h);
        f->SetOctaveCount(4);
        f->SetGain(0.38f);
        auto w = FastNoise::New<FastNoise::DomainWarpGradient>();
        w->SetSource(f);
        w->SetScale(125.0f);
        w->SetWarpAmplitude(28.0f);
        m_hills = w;
    }
    m_hilly = fbm(1100.0f, 2, 0.5f);
    m_plateau = fbm(1700.0f, 2, 0.45f);
    m_gap = fbm(1300.0f, 1, 0.5f);
    {
        auto base = FastNoise::New<FastNoise::Simplex>();
        base->SetScale(460.0f);
        auto r = FastNoise::New<FastNoise::FractalRidged>();
        r->SetSource(base);
        r->SetOctaveCount(3);
        r->SetGain(0.5f);
        auto w = FastNoise::New<FastNoise::DomainWarpGradient>();
        w->SetSource(r);
        w->SetScale(240.0f);
        w->SetWarpAmplitude(70.0f);
        m_ridge = w;
    }
    const auto warped = [](float scale, float warpScale, float amp) {
        auto base = FastNoise::New<FastNoise::Simplex>();
        base->SetScale(scale);
        auto w = FastNoise::New<FastNoise::DomainWarpGradient>();
        w->SetSource(base);
        w->SetScale(warpScale);
        w->SetWarpAmplitude(amp);
        return w;
    };
    m_duneA = warped(78.0f, 260.0f, 34.0f);
    m_duneB = warped(121.0f, 310.0f, 46.0f);
    m_swell = fbm(420.0f, 2, 0.4f);
    m_tempBand = fbm(11000.0f, 2, 0.4f);
    m_wetNoise = fbm(1900.0f, 2, 0.45f);
}

float Terrain3::continentAt(float x, float z) const { return m_cont->GenSingle2D(x, z, m_seed + 301); }

// Each plate slides its own way. The velocity field is a soft blend of the plates around a point,
// so its convergence is a smooth function everywhere, junctions included: squeezed ground rises,
// stretched ground sinks.
Terrain3::Plates Terrain3::platesAt(float x, float z) const {
    const float qx = x + m_warpX->GenSingle2D(x, z, m_seed + 311) * 1000.0f;
    const float qz = z + m_warpZ->GenSingle2D(x, z, m_seed + 313) * 1000.0f;
    const int cx = int(floorDivF(qx, kPlateCell)), cz = int(floorDivF(qz, kPlateCell));

    struct Site {
        float dist, ux, uz, vx, vz;
    };
    Site sites[25];
    int count = 0;
    float nearest = 1e30f;
    for (int j = -2; j <= 2; ++j)
        for (int i = -2; i <= 2; ++i) {
            const int sx = cx + i, sz = cz + j;
            const std::uint32_t h = hash2(sx, sz, std::uint32_t(m_seed) + 7u);
            const float px = (float(sx) + 0.15f + 0.7f * hash01(h)) * kPlateCell;
            const float pz = (float(sz) + 0.15f + 0.7f * hash01(h >> 16)) * kPlateCell;
            const float dx = qx - px, dz = qz - pz;
            const float dist = std::sqrt(dx * dx + dz * dz);
            const std::uint32_t h2 = hash2(sx, sz, std::uint32_t(m_seed) + 13u);
            const float ang = hash01(h2) * 6.2831853f;
            const float speed = 0.55f + 0.45f * hash01(h2 >> 16);
            const float inv = 1.0f / std::max(dist, 1.0f);
            sites[count++] = {dist, dx * inv, dz * inv, std::cos(ang) * speed, std::sin(ang) * speed};
            nearest = std::min(nearest, dist);
        }

    const auto squeeze = [&](float soft) {
        float sw = 0.0f, swx = 0.0f, swz = 0.0f, svx = 0.0f, svz = 0.0f, div = 0.0f;
        for (int k = 0; k < count; ++k) {
            const Site& s = sites[k];
            const float e = (s.dist - nearest) / soft;
            if (e > 10.0f)
                continue;
            const float w = std::exp(-e);
            const float dwx = -w / soft * s.ux, dwz = -w / soft * s.uz;
            sw += w;
            swx += dwx;
            swz += dwz;
            svx += w * s.vx;
            svz += w * s.vz;
            div += dwx * s.vx + dwz * s.vz;
        }
        return (div * sw - (svx * swx + svz * swz)) / (sw * sw) * soft * 2.5f;
    };
    const float n = squeeze(kNarrow), w = squeeze(kWide);
    return {std::max(-n, 0.0f), std::max(-w, 0.0f), std::max(n, 0.0f)};
}

std::shared_ptr<const Terrain3::PlateTile> Terrain3::plateTile(int tx, int tz) const {
    auto& map = m_plateTiles;
    const std::int64_t key = tileKey(tx, tz);
    auto& slot = map.slots[tileBucket(tx, tz)];
    {
        std::lock_guard<std::mutex> g(map.lock);
        if (slot.key == key && slot.tile)
            return slot.tile;
    }
    auto t = std::make_shared<PlateTile>();
    for (int j = 0; j < kPlateN; ++j)
        for (int i = 0; i < kPlateN; ++i) {
            const Plates p = platesAt(float(tx * kTile + (i - 1) * kPlateStep), float(tz * kTile + (j - 1) * kPlateStep));
            const std::size_t k = std::size_t(j) * kPlateN + i;
            t->narrow[k] = p.narrow;
            t->wide[k] = p.wide;
            t->rift[k] = p.rift;
        }
    std::lock_guard<std::mutex> g(map.lock);
    slot.key = key;
    slot.tile = t;
    return t;
}

std::shared_ptr<const Terrain3::ClimateTile> Terrain3::climateTile(int tx, int tz) const {
    auto& map = m_climateTiles;
    const std::int64_t key = tileKey(tx, tz);
    auto& slot = map.slots[tileBucket(tx, tz)];
    {
        std::lock_guard<std::mutex> g(map.lock);
        if (slot.key == key && slot.tile)
            return slot.tile;
    }
    auto t = std::make_shared<ClimateTile>();
    constexpr int kN = kClimateN;
    const float ox = float(tx * kTile - kClimateStep), oz = float(tz * kTile - kClimateStep);
    std::array<float, kN * kN> band, wet;
    m_tempBand->GenUniformGrid2D(band.data(), ox, oz, kN, kN, float(kClimateStep), float(kClimateStep), m_seed + 321);
    m_wetNoise->GenUniformGrid2D(wet.data(), ox, oz, kN, kN, float(kClimateStep), float(kClimateStep), m_seed + 323);
    // Walk upwind: sea adds moisture, and every high range crossed on the way wrings it out, the
    // nearest counting most. So the coast facing the wind is wet, the far side of a range is dry,
    // and the heart of a continent is dry.
    constexpr float kReach[] = {350.0f, 900.0f, 1700.0f, 2800.0f, 4400.0f, 6500.0f};
    constexpr float kSeaWeight[] = {0.34f, 0.26f, 0.20f, 0.14f, 0.10f, 0.08f};
    constexpr float kRangeWeight[] = {0.50f, 0.50f, 0.42f, 0.32f, 0.22f, 0.14f};
    std::array<float, kN * kN> sea{}, shadow{}, buf;
    for (int r = 0; r < 6; ++r) {
        m_cont->GenUniformGrid2D(buf.data(), ox - kWindX * kReach[r], oz - kWindZ * kReach[r], kN, kN, float(kClimateStep),
                                 float(kClimateStep), m_seed + 301);
        for (std::size_t k = 0; k < buf.size(); ++k)
            sea[k] += smooth01(0.02f, -0.12f, buf[k] - kCoastField) * kSeaWeight[r];
    }
    // Slow and very smooth, so worked out on every other point and blended between them. Copying
    // one value over a 2 x 2 block instead made moisture a staircase, and biome borders ran along
    // its steps in straight lines 128 blocks apart.
    const auto shadowAt = [&](int i, int j) {
        const float px = ox + float(i * kClimateStep), pz = oz + float(j * kClimateStep);
        float s = 0.0f;
        for (int r = 0; r < 6; ++r) {
            const Plates p = platesAt(px - kWindX * kReach[r], pz - kWindZ * kReach[r]);
            s += std::min(p.narrow * 0.7f + p.wide * 0.5f, 1.2f) * kRangeWeight[r];
        }
        return s;
    };
    for (int j = 0; j < kN; j += 2)
        for (int i = 0; i < kN; i += 2)
            shadow[std::size_t(j) * kN + i] = shadowAt(i, j);
    // kN is even, so the last row and column have no point beyond them: worked out directly
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i) {
            if (i % 2 == 0 && j % 2 == 0)
                continue;
            if (i == kN - 1 || j == kN - 1) {
                shadow[std::size_t(j) * kN + i] = shadowAt(i, j);
                continue;
            }
            const auto at = [&](int a, int b) { return shadow[std::size_t(b) * kN + a]; };
            float v;
            if (j % 2 == 0)
                v = 0.5f * (at(i - 1, j) + at(i + 1, j));
            else if (i % 2 == 0)
                v = 0.5f * (at(i, j - 1) + at(i, j + 1));
            else
                v = 0.25f * (at(i - 1, j - 1) + at(i + 1, j - 1) + at(i - 1, j + 1) + at(i + 1, j + 1));
            shadow[std::size_t(j) * kN + i] = v;
        }
    for (std::size_t k = 0; k < t->temp.size(); ++k) {
        t->temp[k] = std::clamp(0.60f + band[k] * 0.55f, 0.0f, 1.0f);
        t->moisture[k] = std::clamp(0.08f + sea[k] * 1.05f - shadow[k] * 0.62f + wet[k] * 0.28f, 0.0f, 1.0f);
    }
    std::lock_guard<std::mutex> g(map.lock);
    slot.key = key;
    slot.tile = t;
    return t;
}

namespace {
constexpr int kNbX[8] = {1, -1, 0, 0, 1, 1, -1, -1};
constexpr int kNbZ[8] = {0, 0, 1, -1, 1, -1, 1, -1};
constexpr int kMinRiver = 3; // nodes draining through before a stream shows
} // namespace

std::shared_ptr<const Terrain3::RiverTile> Terrain3::riverTile(int tx, int tz) const {
    auto& map = m_riverTiles;
    const std::int64_t key = tileKey(tx, tz);
    auto& slot = map.slots[tileBucket(tx, tz)];
    {
        std::unique_lock<std::mutex> g(map.lock);
        for (;;) {
            if (slot.key == key && slot.tile)
                return slot.tile;
            if (std::find(map.inFlight.begin(), map.inFlight.end(), key) == map.inFlight.end())
                break;
            map.built.wait(g); // another thread is building this one
        }
        map.inFlight.push_back(key);
    }
    constexpr int T = kRiverTile, P = kRiverPad, W = T + 2 * P;
    constexpr float S = float(kRiverCell);
    const int i0 = tx * T - P, j0 = tz * T - P;
    std::vector<RiverNode> nodes(std::size_t(W) * W);
    for (int b = 0; b < W; ++b)
        for (int a = 0; a < W; ++a) {
            const std::uint32_t hsh = hash2(i0 + a, j0 + b, std::uint32_t(m_seed) + 71u);
            RiverNode& nd = nodes[std::size_t(b) * W + a];
            nd.x = (float(i0 + a) + 0.5f + (hash01(hsh) - 0.5f) * 0.64f) * S;
            nd.z = (float(j0 + b) + 0.5f + (hash01(hsh >> 16) - 0.5f) * 0.64f) * S;
        }
    // The land's broad shape under all of them, from one coarse pass, read between its points.
    constexpr int L = 64;
    const int lx0 = int(std::floor(float(i0) * S / float(L))) * L - L, lz0 = int(std::floor(float(j0) * S / float(L))) * L - L;
    const int ln = (W * kRiverCell) / L + 4;
    std::vector<Sample> land(std::size_t(ln) * ln);
    gridImpl(lx0, lz0, ln, L, land.data(), true);
    for (RiverNode& nd : nodes) {
        const float fx = (nd.x - float(lx0)) / float(L), fz = (nd.z - float(lz0)) / float(L);
        const int ix = std::clamp(int(fx), 0, ln - 2), iz = std::clamp(int(fz), 0, ln - 2);
        const float tx2 = fx - float(ix), tz2 = fz - float(iz);
        const auto at = [&](int x, int z) { return land[std::size_t(z) * ln + x].h; };
        nd.h = glm::mix(glm::mix(at(ix, iz), at(ix + 1, iz), tx2), glm::mix(at(ix, iz + 1), at(ix + 1, iz + 1), tx2), tz2);
    }
    const auto node = [&](int a, int b) -> RiverNode& { return nodes[std::size_t(b) * W + a]; };
    // Each drains to its lowest neighbour. One lower than all of them is a basin and fills with a lake
    // up to the lowest way out.
    for (int b = 0; b < W; ++b)
        for (int a = 0; a < W; ++a) {
            RiverNode& nd = node(a, b);
            if (nd.h < float(kSeaLevel))
                continue;
            float best = nd.h, pass = 1e9f;
            for (int k = 0; k < 8; ++k) {
                const int na = a + kNbX[k], nb = b + kNbZ[k];
                if (na < 0 || nb < 0 || na >= W || nb >= W)
                    continue;
                const float hn = node(na, nb).h;
                pass = std::min(pass, hn);
                if (hn < best) {
                    best = hn;
                    nd.down = k;
                }
            }
            if (nd.down < 0 && pass < 1e8f)
                nd.lake = std::min(pass, nd.h + 16.0f);
        }
    // What drains through each node: every node within kRiverReach whose path passes it.
    for (int b = 0; b < W; ++b)
        for (int a = 0; a < W; ++a) {
            int va = a, vb = b;
            for (int step = 0; step <= kRiverReach; ++step) {
                if (std::max(std::abs(va - a), std::abs(vb - b)) > kRiverReach)
                    break;
                ++node(va, vb).acc;
                const int d = node(va, vb).down;
                if (d < 0)
                    break;
                va += kNbX[d];
                vb += kNbZ[d];
                if (va < 0 || vb < 0 || va >= W || vb >= W)
                    break;
            }
        }
    auto t = std::make_shared<RiverTile>();
    for (int b = 0; b < T; ++b)
        for (int a = 0; a < T; ++a) {
            RiverNode nd = node(a + P, b + P);
            // the river sits below the land around it, deeper the more it carries
            nd.level = std::max(float(kSeaLevel), nd.h - (2.5f + float(std::min(nd.acc, 60)) * 0.10f));
            t->nodes[std::size_t(b) * T + a] = nd;
        }
    {
        std::lock_guard<std::mutex> g(map.lock);
        slot.key = key;
        slot.tile = t;
        map.inFlight.erase(std::find(map.inFlight.begin(), map.inFlight.end(), key));
    }
    map.built.notify_all();
    return t;
}

void Terrain3::collectRivers(float x0, float z0, float x1, float z1, std::vector<Reach>& reaches, std::vector<Lake>& lakes) const {
    constexpr float kMargin = 300.0f;
    const int i0 = int(std::floor((x0 - kMargin) / float(kRiverCell))) - 1;
    const int i1 = int(std::floor((x1 + kMargin) / float(kRiverCell))) + 1;
    const int j0 = int(std::floor((z0 - kMargin) / float(kRiverCell))) - 1;
    const int j1 = int(std::floor((z1 + kMargin) / float(kRiverCell))) + 1;
    std::shared_ptr<const RiverTile> last;
    int ltx = INT_MIN, ltz = INT_MIN;
    const auto nodeAt = [&](int i, int j) -> const RiverNode& {
        const int tx = int(std::floor(float(i) / float(kRiverTile))), tz = int(std::floor(float(j) / float(kRiverTile)));
        if (tx != ltx || tz != ltz) {
            last = riverTile(tx, tz);
            ltx = tx;
            ltz = tz;
        }
        return last->nodes[std::size_t(j - tz * kRiverTile) * kRiverTile + std::size_t(i - tx * kRiverTile)];
    };
    for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i) {
            const RiverNode nd = nodeAt(i, j);
            if (nd.acc < kMinRiver)
                continue;
            if (nd.down < 0) {
                if (nd.lake > nd.h + 1.0f)
                    lakes.push_back({nd.x, nd.z, std::min(110.0f + 14.0f * std::sqrt(float(nd.acc)), float(kRiverCell) * 0.55f), nd.lake});
                continue;
            }
            const RiverNode dn = nodeAt(i + kNbX[nd.down], j + kNbZ[nd.down]);
            const float size = std::sqrt(float(std::min(nd.acc, 80)));
            const float half = 1.5f + 1.6f * size;
            reaches.push_back({nd.x, nd.z, dn.x, dn.z, nd.level, std::min(dn.level, nd.level), half,
                               std::min(1.5f + 0.75f * size, 7.0f), half * 2.5f + 18.0f});
        }
}

void Terrain3::grid(int x0, int z0, int n, int step, Sample* out) const { gridImpl(x0, z0, n, step, out, false); }

void Terrain3::gridImpl(int x0, int z0, int n, int step, Sample* out, bool smooth) const {
    // Rivers first: building a river tile runs this function too, on the same scratch buffers. A smooth
    // pass is that build, and must leave the list its caller is filling alone.
    static thread_local std::vector<Reach>& riverList = *new std::vector<Reach>();
    static thread_local std::vector<Lake>& lakeList = *new std::vector<Lake>();
    static const std::vector<Reach> noReaches;
    static const std::vector<Lake> noLakes;
    if (!smooth) {
        riverList.clear();
        lakeList.clear();
        collectRivers(float(x0), float(z0), float(x0 + (n - 1) * step), float(z0 + (n - 1) * step), riverList, lakeList);
    }
    const std::vector<Reach>& reaches = smooth ? noReaches : riverList;
    const std::vector<Lake>& lakes = smooth ? noLakes : lakeList;

    // A ring around the output for slopes.
    const int g = step == 1 ? 4 : 1;
    const int N = n + 2 * g;
    const std::size_t NN = std::size_t(N) * std::size_t(N);
    const float gx0 = float(x0 - g * step), gz0 = float(z0 - g * step), fs = float(step);

    // never destroyed on purpose: see computeConnectivity
    static thread_local std::vector<float>& scratch = *new std::vector<float>();
    scratch.resize(NN * 10);
    float* hills = scratch.data();
    float* hilly = hills + NN;
    float* plateau = hilly + NN;
    float* gap = plateau + NN;
    float* ridge = gap + NN;
    float* mountain = ridge + NN;
    float* cVal = mountain + NN;
    float* cGrad = cVal + NN;
    float* wide = cGrad + NN;
    float* riftV = wide + NN;

    if (smooth)
        std::fill(hills, hills + NN, 0.0f);
    else
        m_hills->GenUniformGrid2D(hills, gx0, gz0, N, N, fs, fs, m_seed);
    m_hilly->GenUniformGrid2D(hilly, gx0, gz0, N, N, fs, fs, m_seed + 303);
    m_plateau->GenUniformGrid2D(plateau, gx0, gz0, N, N, fs, fs, m_seed + 305);
    m_gap->GenUniformGrid2D(gap, gx0, gz0, N, N, fs, fs, m_seed + 307);
    m_ridge->GenUniformGrid2D(ridge, gx0, gz0, N, N, fs, fs, m_seed + 309);

    // The continent field is smooth: on a coarse lattice for the points close together. Its slope,
    // over 16 blocks either side, says how far the coast is.
    {
        const int L = step == 1 ? 8 : step;
        const int span = L >= 16 ? 1 : 16 / L;
        // Padded past the slope's reach, so every node a point reads has the same centred slope
        // whatever grid it is part of. Three nodes was short of it at steps 2 and 4: straight seams
        // in the sea floor wherever one worker's part of a big grid met the next.
        const int pad = std::max(3, span + 2);
        const int lx0 = int(std::floor(gx0 / float(L))) * L - pad * L, lz0 = int(std::floor(gz0 / float(L))) * L - pad * L;
        const int ln = ((N - 1) * step) / L + 2 * pad + 2;
        const std::size_t ll = std::size_t(ln) * ln;
        static thread_local std::vector<float>& lat = *new std::vector<float>();
        lat.resize(ll * (5 + kLookDirs));
        float* v = lat.data();
        float* lgx = v + ll;
        float* lgz = lgx + ll;
        float* lookD = lgz + ll;
        float* lookAll = lookD + ll;
        m_cont->GenUniformGrid2D(v, float(lx0), float(lz0), ln, ln, float(L), float(L), m_seed + 301);
        for (int k = 0; k < kLookDirs; ++k)
            m_cont->GenUniformGrid2D(lookAll + ll * std::size_t(k), float(lx0 + kLookX[k]), float(lz0 + kLookZ[k]), ln, ln,
                                     float(L), float(L), m_seed + 301);
        for (int j = 0; j < ln; ++j)
            for (int i = 0; i < ln; ++i) {
                const std::size_t k = std::size_t(j) * ln + i;
                const int a0 = std::max(i - span, 0), a1 = std::min(i + span, ln - 1);
                const int b0 = std::max(j - span, 0), b1 = std::min(j + span, ln - 1);
                lgx[k] = (v[std::size_t(j) * ln + a1] - v[std::size_t(j) * ln + a0]) / float((a1 - a0) * L);
                lgz[k] = (v[std::size_t(b1) * ln + i] - v[std::size_t(b0) * ln + i]) / float((b1 - b0) * L);
                // A secant hundreds of blocks long sees the field curve where the local slope cannot.
                const float c = v[k] - kCoastField;
                float best = 2000.0f;
                for (int dir = 0; dir < kLookDirs; ++dir) {
                    const float drop = c - (lookAll[ll * std::size_t(dir) + k] - kCoastField);
                    if (drop > 1e-4f)
                        best = std::min(best, std::sqrt(float(kLookX[dir] * kLookX[dir] + kLookZ[dir] * kLookZ[dir])) * c / drop);
                }
                lookD[k] = best;
            }
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const float fx = (gx0 + float(i * step) - float(lx0)) / float(L) - 1.0f;
                const float fz = (gz0 + float(j * step) - float(lz0)) / float(L) - 1.0f;
                const std::size_t k = std::size_t(j) * N + i;
                cVal[k] = cubic2(v, ln, fx, fz);
                const float ax = cubic2(lgx, ln, fx, fz), az = cubic2(lgz, ln, fx, fz);
                const float c = cVal[k] - kCoastField;
                const float grad = std::max(std::sqrt(ax * ax + az * az), 1.0f / 2500.0f);
                float d = std::clamp(c / grad, -2000.0f, 2000.0f);
                if (c > 0.0f) {
                    const float look = cubic2(lookD, ln, fx, fz);
                    d = d > look ? look : glm::mix(d, look, smooth01(60.0f, 200.0f, d));
                }
                cGrad[k] = d; // from here on this holds the distance from the coast
            }
    }

    // The plates, then the mass of each point's mountains. Everything that climbs is faded in with
    // distance from the coast, so a range always has country in front of it.
    {
        std::shared_ptr<const PlateTile> tile;
        int ctx = INT_MIN, ctz = INT_MIN;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const std::size_t k = std::size_t(j) * N + i;
                const int wx = int(gx0) + i * step, wz = int(gz0) + j * step;
                const int tx = int(std::floor(float(wx) / float(kTile))), tz = int(std::floor(float(wz) / float(kTile)));
                if (tx != ctx || tz != ctz) {
                    tile = plateTile(tx, tz);
                    ctx = tx;
                    ctz = tz;
                }
                const float fx = float(wx - tx * kTile) / float(kPlateStep);
                const float fz = float(wz - tz * kTile) / float(kPlateStep);
                const float un = cubic2(tile->narrow.data(), kPlateN, fx, fz);
                const float uw = cubic2(tile->wide.data(), kPlateN, fx, fz);
                wide[k] = uw;
                riftV[k] = cubic2(tile->rift.data(), kPlateN, fx, fz);

                const float d = cGrad[k];
                const float reach = smooth01(220.0f, 1500.0f, d);
                const float on = 0.40f + 0.85f * smooth01(-0.42f, 0.28f, gap[k]);
                const float r = std::clamp(ridge[k] * 0.5f + 0.5f, 0.0f, 1.0f);
                // The squeeze sets how high a range can get; the ridgelines decide what actually
                // stands up there, valleys cut deep between them.
                const float core = 1.0f - std::exp(-std::max(un, 0.0f) * 1.7f);
                const float foot = std::pow(std::max(uw, 0.0f), 1.4f) * 44.0f * (0.65f + 0.35f * on) * reach;
                mountain[k] = foot + core * 170.0f * on * reach * (0.16f + 1.22f * std::pow(r, 1.5f));
                mountain[k] += hills[k] * mountain[k] * 0.28f;
                // no faster than about one up for three in, whatever the plates say
                const float cap = 0.34f * std::max(d - 120.0f, 0.0f);
                const float m = mountain[k];
                mountain[k] = std::max(0.5f * (m + cap - std::sqrt((m - cap) * (m - cap) + 64.0f)), 0.0f);
            }
    }

    // The sea, the coast and the land's slow rise: version 2's profile, because the sea is right.
    static constexpr float kDX[] = {-700.0f, -420.0f, -260.0f, -30.0f, 0.0f, 14.0f, 120.0f, 400.0f};
    static constexpr float kDY[] = {24.0f, 30.0f, 48.0f, 58.0f, 62.0f, 64.0f, 71.0f, 85.0f};
    static constexpr float kCX[] = {-0.40f, -0.20f, 0.0f, 0.20f, 0.45f, 0.80f};
    static constexpr float kCY[] = {20.0f, 24.0f, 72.0f, 85.0f, 100.0f, 110.0f};

    std::shared_ptr<const ClimateTile> ctile;
    int ccx = INT_MIN, ccz = INT_MIN;
    bool anyDesert = false;
    static thread_local std::vector<float>& climate = *new std::vector<float>();
    climate.resize(std::size_t(n) * n * 3);
    float* tempV = climate.data();
    float* wetV = tempV + std::size_t(n) * n;
    float* desertV = wetV + std::size_t(n) * n;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const int wx = x0 + i * step, wz = z0 + j * step;
            const std::size_t kg = std::size_t(j + g) * N + i + g;
            // read a little off to the side by the local noise, so no edge is a smooth contour. The
            // tile is the one the bent point lands in: picked by the straight one, a read near its
            // edge ran off the tile's grid and every tile border was a straight seam in the biomes.
            const float qx = float(wx) + gap[kg] * 180.0f + hills[kg] * 60.0f;
            const float qz = float(wz) + plateau[kg] * 180.0f - hills[kg] * 60.0f;
            const int tx = int(std::floor(qx / float(kTile))), tz = int(std::floor(qz / float(kTile)));
            if (tx != ccx || tz != ccz) {
                ctile = climateTile(tx, tz);
                ccx = tx;
                ccz = tz;
            }
            const float fx = (qx - float(tx * kTile)) / float(kClimateStep);
            const float fz = (qz - float(tz * kTile)) / float(kClimateStep);
            const std::size_t k = std::size_t(j) * n + i;
            tempV[k] = cubic2(ctile->temp.data(), kClimateN, fx, fz);
            wetV[k] = cubic2(ctile->moisture.data(), kClimateN, fx, fz);
            const float d = cGrad[std::size_t(j + g) * N + i + g];
            desertV[k] = smooth01(0.56f, 0.70f, tempV[k]) * smooth01(0.34f, 0.20f, wetV[k]) * smooth01(40.0f, 200.0f, d);
            anyDesert = anyDesert || desertV[k] > 0.0f;
        }

    static thread_local std::vector<float>& dunes = *new std::vector<float>();
    float* duneA = nullptr;
    float* duneB = nullptr;
    float* swell = nullptr;
    if (anyDesert) {
        dunes.resize(std::size_t(n) * n * 3);
        duneA = dunes.data();
        duneB = duneA + std::size_t(n) * n;
        swell = duneB + std::size_t(n) * n;
        m_duneA->GenUniformGrid2D(duneA, float(x0), float(z0), n, n, fs, fs, m_seed + 331);
        m_duneB->GenUniformGrid2D(duneB, float(x0), float(z0), n, n, fs, fs, m_seed + 333);
        m_swell->GenUniformGrid2D(swell, float(x0), float(z0), n, n, fs, fs, m_seed + 335);
    }

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const std::size_t k = std::size_t(j + g) * N + i + g;
            const std::size_t o = std::size_t(j) * n + i;
            const float wx = float(x0 + i * step), wz = float(z0 + j * step);
            const float c = cVal[k] - kCoastField;
            const float d = cGrad[k];
            const float desert = desertV[o];

            const float nearCoast = curve(kDX, kDY, 8, d);
            const float farCoast = curve(kCX, kCY, 6, c);
            const float far = glm::mix(nearCoast, farCoast, smooth01(350.0f, 650.0f, std::abs(d)));
            float h = c >= 0.0f ? smoothMax(nearCoast, far, 6.0f) : -smoothMax(-nearCoast, -far, 6.0f);

            const float inland = smooth01(12.0f, 220.0f, d);
            const float rolling = smooth01(-0.25f, 0.35f, hilly[k]);
            float amp = 1.2f + inland * (4.0f + 14.0f * rolling);
            amp *= 1.0f - 0.65f * desert;
            h += (hills[k] + 0.15f * inland) * amp;
            if (d < 0.0f)
                h += hills[k] * 4.0f * smooth01(-40.0f, -300.0f, d);

            // Plateaus: a step in a broad field, so a flat top with a ramp up to it.
            const float away = smooth01(500.0f, 1100.0f, d) * (1.0f - smooth01(0.1f, 0.5f, wide[k]));
            h += smooth01(0.30f, 0.40f, plateau[k]) * (20.0f + 12.0f * rolling) * away * (1.0f - 0.5f * desert);
            // stretched ground sinks into wide lowlands
            h -= riftV[k] * 16.0f * smooth01(500.0f, 1200.0f, d);
            const float broad = h - (hills[k] + 0.15f * inland) * amp + mountain[k];

            float m = mountain[k];
            if (m > 2.0f && !smooth) {
                const std::size_t kx0 = k - std::size_t(g), kx1 = k + std::size_t(g);
                const std::size_t kz0 = k - std::size_t(g) * N, kz1 = k + std::size_t(g) * N;
                const float span = float(2 * g * step);
                const float sx = (mountain[kx1] - mountain[kx0]) / span, sz = (mountain[kz1] - mountain[kz0]) / span;
                const float steep = std::sqrt(sx * sx + sz * sz);
                const float strength = std::min(m * 0.24f + 1.0f, 38.0f) * smooth01(0.01f, 0.10f, steep);
                if (strength > 0.05f)
                    m += erode(wx, wz, sx, sz, strength, step, std::uint32_t(m_seed) + 41u);
                m = std::max(m, 0.0f);
            }
            h += m;

            const std::uint32_t ms = std::uint32_t(m_seed);
            // Not every range is the same country. A slow field picks, range by range: crags on the
            // upper slopes, the plain eroded ridges, or a range stepped in benches and cliff bands.
            if (m > 4.0f && !smooth) {
                const float jag = smooth01(0.56f, 0.74f, valueNoise(wx / 2600.0f, wz / 2600.0f, ms + 501u));
                const float terr = smooth01(0.60f, 0.78f, valueNoise(wx / 3100.0f, wz / 3100.0f, ms + 503u)) * (1.0f - jag);
                if (jag > 0.0f) {
                    // fewer octaves where the far horizon samples coarsely, or they shimmer
                    const int octaves = step <= 16 ? 2 : 1;
                    float crag = 0.0f, amp = 1.0f, f = 1.0f / 64.0f;
                    for (int oc = 0; oc < octaves; ++oc, amp *= 0.5f, f *= 2.1f) {
                        const float v = 1.0f - std::abs(valueNoise(wx * f, wz * f, ms + 505u + std::uint32_t(oc)) * 2.0f - 1.0f);
                        crag += v * v * amp;
                    }
                    h += crag * 15.0f * jag * smooth01(25.0f, 90.0f, m);
                }
                if (terr > 0.0f) {
                    const float stepH = 13.0f + 7.0f * valueNoise(wx / 400.0f, wz / 400.0f, ms + 509u);
                    const float t = m / stepH, frac = t - std::floor(t);
                    const float stepped = (std::floor(t) + smooth01(0.68f, 1.0f, frac)) * stepH;
                    h += (stepped - m) * terr * smooth01(15.0f, 50.0f, m);
                }
            }

            // Karst: in warm wet country well away from the coast, towers of rock standing straight up
            // out of the forest, tens of blocks high and a few dozen apart.
            const float karst = smooth01(0.58f, 0.68f, tempV[o]) * smooth01(0.52f, 0.64f, wetV[o]) *
                                smooth01(300.0f, 700.0f, d) * (1.0f - desert) * (1.0f - smooth01(10.0f, 40.0f, m));
            if (!smooth && karst > 0.0f && step <= 16) {
                constexpr float kCell = 70.0f;
                const int cx = int(std::floor(wx / kCell)), cz = int(std::floor(wz / kCell));
                float tower = 0.0f;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const std::uint32_t th = hash2(cx + dx, cz + dz, ms + 521u);
                        if ((th & 255u) > 120u)
                            continue;
                        const float tx = (float(cx + dx) + 0.2f + 0.6f * hash01(th >> 8)) * kCell;
                        const float tz = (float(cz + dz) + 0.2f + 0.6f * hash01(hash2(cx + dx, cz + dz, ms + 523u))) * kCell;
                        const float r = 6.0f + 8.0f * hash01(hash2(cx + dx, cz + dz, ms + 525u));
                        const float tall = 30.0f + 60.0f * hash01(hash2(cx + dx, cz + dz, ms + 527u));
                        const float ddx = wx - tx, ddz = wz - tz;
                        const float dist = std::sqrt(ddx * ddx + ddz * ddz) * (0.85f + 0.3f * valueNoise(wx * 0.18f, wz * 0.18f, ms + 529u));
                        const float wall = 1.0f - smooth01(r - 2.0f, r + 1.5f, dist);
                        tower = std::max(tower, wall * (tall + 5.0f * valueNoise(wx * 0.25f, wz * 0.25f, ms + 531u)));
                    }
                h += tower * karst;
            }

            // Volcanoes: now and then a lone cone with a crater at the top and a lake in it, an island
            // where one rises out of the sea. In absolute heights, so the crater floor and its lake are
            // level whatever the country under it, and in the smoothed land too, so rivers run off it.
            float volcanic = 0.0f, craterLake = -1.0f;
            {
                constexpr float kVCell = 2600.0f;
                const int vx = int(std::floor(wx / kVCell)), vz = int(std::floor(wz / kVCell));
                const std::uint32_t vh = hash2(vx, vz, ms + 541u);
                if ((vh & 255u) < 70u) {
                    const float cxw = (float(vx) + 0.3f + 0.4f * hash01(vh >> 8)) * kVCell;
                    const float czw = (float(vz) + 0.3f + 0.4f * hash01(hash2(vx, vz, ms + 543u))) * kVCell;
                    const float rb = 300.0f + 160.0f * hash01(hash2(vx, vz, ms + 545u));
                    const float hv = 110.0f + 70.0f * hash01(hash2(vx, vz, ms + 547u));
                    const float rc = 36.0f + 22.0f * hash01(hash2(vx, vz, ms + 549u));
                    const float ddx = wx - cxw, ddz = wz - czw;
                    const float dist = std::sqrt(ddx * ddx + ddz * ddz);
                    if (dist < rb) {
                        constexpr float kBase = 70.0f;
                        const auto cone = [&](float r) { return hv * std::pow(1.0f - r / rb, 1.7f); };
                        const float rim = kBase + cone(rc);
                        // gullies running down the flanks: noise round a circle of directions, which has
                        // no seam where an angle would wrap
                        const float ux = ddx / std::max(dist, 1.0f), uz = ddz / std::max(dist, 1.0f);
                        const float gully = smooth ? 0.0f
                                                   : (valueNoise(40.0f + ux * 9.0f + dist * 0.004f, 40.0f + uz * 9.0f, ms + 551u) - 0.5f) *
                                                         10.0f * smooth01(0.0f, 0.4f, dist / rb);
                        if (dist < rc) {
                            h = rim - 0.45f * rc * (1.0f - (dist / rc) * (dist / rc));
                            if (dist < rc * 0.9f)
                                craterLake = rim - 8.0f;
                        } else {
                            // standing on the ground under it at the foot, on its own fixed heights by
                            // the top: a fixed foot rose out of the sea floor as a circular cliff
                            const float onGround = h + cone(dist) + gully;
                            const float fixedTop = std::max(h, kBase + cone(dist) + gully);
                            h = glm::mix(onGround, fixedTop, smooth01(rb * 0.75f, rb * 0.35f, dist));
                        }
                        volcanic = 1.0f - smooth01(rb * 0.45f, rb, dist);
                    }
                }
            }

            if (desert > 0.0f && duneA != nullptr && !smooth) {
                // two crest fields crossing, on long swells: narrow crests, wide troughs
                const float a = 1.0f - std::abs(std::clamp(duneA[o], -1.0f, 1.0f));
                const float b = 1.0f - std::abs(std::clamp(duneB[o], -1.0f, 1.0f));
                h += (a * a * 5.0f + b * b * 7.0f + (swell[o] * 0.5f + 0.5f) * 11.0f) * desert;
            }

            // a coast that is too smooth quantises into long straight terraces
            const float shore = 1.0f - smooth01(1.5f, 6.0f, std::abs(h - float(kSeaLevel)));
            if (shore > 0.0f && !smooth) {
                const std::uint32_t salt = std::uint32_t(m_seed) + 211u;
                h += ((valueNoise(wx * 0.04f, wz * 0.04f, salt) - 0.5f) * 2.2f +
                      (valueNoise(wx * 0.17f, wz * 0.17f, salt + 2u) - 0.5f) * 0.8f) * shore;
            }
            // the top of the world is a long way up: compress, never clip
            if (h > 300.0f)
                h = 300.0f + (h - 300.0f) * 0.4f;
            // A hollow inland that dips under the sea is not a lake, only a pond the sea level leaves
            // behind. It fills to a flat just above the water instead.
            if (!smooth && d > 24.0f && h < float(kSeaLevel) + 0.6f)
                h = glm::mix(h, float(kSeaLevel) + 0.6f + (valueNoise(wx * 0.09f, wz * 0.09f, std::uint32_t(m_seed) + 421u)) * 0.8f,
                             smooth01(24.0f, 48.0f, d));

            float water = -1.0f;
            // the sea floor keeps its shape: a river ends where it meets the sea
            if ((!reaches.empty() || !lakes.empty()) && h >= float(kSeaLevel) - 1.0f) {
                // A river wanders, wide loops in low country and tighter ones between
                const std::uint32_t ms = std::uint32_t(m_seed) + 401u;
                const float qx = wx + (valueNoise(wx / 270.0f, wz / 270.0f, ms) - 0.5f) * 150.0f +
                                 (valueNoise(wx / 85.0f, wz / 85.0f, ms + 2u) - 0.5f) * 40.0f;
                const float qz = wz + (valueNoise(wx / 270.0f, wz / 270.0f, ms + 4u) - 0.5f) * 150.0f +
                                 (valueNoise(wx / 85.0f, wz / 85.0f, ms + 6u) - 0.5f) * 40.0f;
                float carved = h, bankUp = -1e9f;
                bool channel = false;
                for (const Reach& r : reaches) {
                    const float ex = r.bx - r.ax, ez = r.bz - r.az;
                    const float t = std::clamp(((qx - r.ax) * ex + (qz - r.az) * ez) / std::max(ex * ex + ez * ez, 1.0f), 0.0f, 1.0f);
                    const float dx = qx - (r.ax + ex * t), dz = qz - (r.az + ez * t);
                    const float dist = std::sqrt(dx * dx + dz * dz);
                    if (dist >= r.valley)
                        continue;
                    const float level = glm::mix(r.levelA, r.levelB, t);
                    const float bank = std::floor(level) + 1.0f;
                    if (dist < r.half) {
                        const float u = dist / r.half;
                        // Never above the ground it runs through. The level is a straight line between
                        // two nodes a long way apart, read off the smoothed land, and it passed over
                        // every dip between them: the water went across on walls, along the beaches too.
                        const float lv = std::min(level, h - 1.0f);
                        carved = std::min(carved, lv - r.depth * (1.0f - u * u) - 0.6f);
                        // a desert keeps the bed and loses the water
                        if (desert < 0.6f)
                            water = std::max(water, std::floor(lv));
                        channel = true;
                    } else {
                        // the valley sides come down to the banks
                        if (h > bank)
                            carved = std::min(carved, bank + (h - bank) * smooth01(r.half, r.valley, dist));
                        // and low ground beside the water comes up a block to meet it, never more
                        if (dist < r.half + 3.0f)
                            bankUp = std::max(bankUp, std::min(bank, h + 1.0f));
                    }
                }
                if (!channel && bankUp > carved)
                    carved = bankUp;
                h = carved;
                for (const Lake& l : lakes) {
                    const float dx = wx - l.x, dz = wz - l.z;
                    // a shore that wanders in and out, not a circle
                    const float radius = l.radius * (0.70f + 0.50f * valueNoise(wx / 64.0f, wz / 64.0f, std::uint32_t(m_seed) + 431u));
                    const float dist = std::sqrt(dx * dx + dz * dz);
                    if (dist >= radius || h >= l.level)
                        continue;
                    // the far edge closes the basin: a rim where the ground would let it out
                    if (dist > radius * 0.82f) {
                        h = std::max(h, std::floor(l.level) + 1.0f);
                        continue;
                    }
                    water = std::max(water, std::floor(l.level));
                }
            }

            if (craterLake > h && craterLake > water)
                water = std::floor(craterLake);
            Sample& s = out[o];
            s.h = h;
            s.volcanic = volcanic;
            s.broad = broad;
            s.coast = d;
            s.desert = desert;
            s.moisture = wetV[o];
            s.temp = tempV[o];
            // cold air comes down the mountain: the warmer the climate the higher the snow
            s.snowLine = 158.0f + (tempV[o] - 0.10f) * 160.0f + ridge[k] * 14.0f + hills[k] * 12.0f;
            s.water = water;
        }
}

} // namespace vx
