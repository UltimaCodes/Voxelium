#include "render/Wildlife.hpp"

#include "core/Log.hpp"
#include "gfx/Shader.hpp"
#include "world/ChunkManager.hpp"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <utility>

namespace vx {

namespace {

constexpr float kPi = 3.14159265f;

// The y of the first solid block at or below yFrom, searching `depth` blocks down. Unloaded
// chunks read as stone, so a spawn point in the void simply finds "ground" at once and is
// rejected by the caller's clearance test.
int floorBelow(const ChunkManager& w, int x, int z, int yFrom, int depth) {
    for (int y = yFrom; y >= yFrom - depth; --y)
        if (blocks::opaque(w.blockAtWorld({x, y, z})))
            return y;
    return INT_MIN;
}

bool isSolidAt(const ChunkManager& w, const glm::dvec3& p) {
    return blocks::opaque(w.blockAtWorld(glm::ivec3(glm::floor(p))));
}
bool isWetAt(const ChunkManager& w, const glm::dvec3& p) {
    return blocks::isWater(w.blockAtWorld(glm::ivec3(glm::floor(p))));
}

const glm::vec3 kButterflyColours[] = {{1.00f, 0.55f, 0.12f}, {1.00f, 0.85f, 0.20f},
                                       {0.40f, 0.60f, 1.00f}, {0.95f, 0.95f, 0.95f},
                                       {0.90f, 0.35f, 0.55f}, {0.60f, 0.40f, 0.90f}};
const glm::vec3 kReefFish[] = {{1.00f, 0.50f, 0.10f}, {1.00f, 0.85f, 0.15f},
                               {0.20f, 0.50f, 1.00f}, {0.90f, 0.25f, 0.60f},
                               {0.10f, 0.80f, 0.70f}};
// The back colour: the belly is worked out from it in wildlife3d.vert (countershading), so these
// lean darker and more saturated than the fish reads overall. The old greys and pale blues were
// the water's own colour and vanished in it.
const glm::vec3 kOpenFish[] = {{0.26f, 0.44f, 0.66f}, {0.40f, 0.52f, 0.24f},
                               {0.90f, 0.42f, 0.28f}, {0.52f, 0.56f, 0.64f}};
const glm::vec3 kLeafColours[] = {{0.25f, 0.55f, 0.15f}, {0.42f, 0.60f, 0.14f},
                                  {0.55f, 0.58f, 0.12f}, {0.80f, 0.45f, 0.10f}};

} // namespace

float Wildlife::rnd() {
    m_rng ^= m_rng << 13;
    m_rng ^= m_rng >> 17;
    m_rng ^= m_rng << 5;
    return float(m_rng & 0xFFFFFFu) / float(0x1000000u);
}

bool Wildlife::init(const std::filesystem::path& shaderDir) {
    m_prog = loadProgram(shaderDir / "wildlife.vert", shaderDir / "wildlife.frag");
    m_prog3d = loadProgram(shaderDir / "wildlife3d.vert", shaderDir / "wildlife3d.frag");
    if (!m_prog || !m_prog3d)
        return false;
    glCreateVertexArrays(1, &m_vao);
    return m_ring.init(std::size_t(kCapacity) * sizeof(GpuSprite));
}

void Wildlife::shutdown() {
    if (m_prog)
        glDeleteProgram(m_prog);
    if (m_prog3d)
        glDeleteProgram(m_prog3d);
    m_prog = m_prog3d = 0;
    m_ring.shutdown();
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    m_vao = 0;
    m_live = 0;
}

Wildlife::Critter* Wildlife::acquire() {
    if (m_live >= kCapacity)
        return nullptr;
    Critter& c = m_pool[m_live++];
    c = Critter{};
    return &c;
}

int Wildlife::count(Kind k) const {
    int n = 0;
    for (int i = 0; i < m_live; ++i)
        n += m_pool[i].kind == k && m_pool[i].life > 1.0f ? 1 : 0;
    return n;
}

void Wildlife::spawnButterfly(const ChunkManager& w, const glm::dvec3& cam) {
    const float ang = rnd() * 2.0f * kPi, r = 8.0f + rnd() * 18.0f;
    const int x = int(std::floor(cam.x + std::cos(ang) * r));
    const int z = int(std::floor(cam.z + std::sin(ang) * r));
    const int y = floorBelow(w, x, z, int(cam.y) + 10, 26);
    if (y == INT_MIN || w.blockAtWorld({x, y, z}) != blocks::Grass)
        return;
    if (blocks::opaque(w.blockAtWorld({x, y + 1, z})) || blocks::opaque(w.blockAtWorld({x, y + 2, z})))
        return;
    Critter* c = acquire();
    if (!c)
        return;
    c->kind = Butterfly;
    c->pos = {double(x) + 0.5, double(y) + 1.6, double(z) + 0.5};
    c->color = kButterflyColours[int(rnd() * 5.999f)] * 1.5f;
    c->size = 0.15f + rnd() * 0.06f;
    c->phase = rnd() * 6.28f;
    c->aux = rnd() * 6.28f;
    c->aux2 = float(y + 1);
}

void Wildlife::spawnFirefly(const ChunkManager& w, const glm::dvec3& cam) {
    const float ang = rnd() * 2.0f * kPi, r = 5.0f + rnd() * 20.0f;
    const int x = int(std::floor(cam.x + std::cos(ang) * r));
    const int z = int(std::floor(cam.z + std::sin(ang) * r));
    const int y = floorBelow(w, x, z, int(cam.y) + 10, 26);
    if (y == INT_MIN || blocks::opaque(w.blockAtWorld({x, y + 2, z})))
        return;
    Critter* c = acquire();
    if (!c)
        return;
    c->kind = Firefly;
    c->pos = {double(x) + 0.5, double(y) + 1.4 + double(rnd()) * 2.0, double(z) + 0.5};
    c->color = glm::vec3(0.85f, 1.0f, 0.30f) * 2.4f;
    c->size = 0.22f;
    c->phase = rnd() * 6.28f;
    c->aux = rnd() * 6.28f;
    c->aux2 = float(y + 1);
    c->aux3 = 1.4f + rnd() * 1.6f; // pulse rate
}

void Wildlife::spawnGull(const ChunkManager&, const glm::dvec3& cam) {
    Critter* c = acquire();
    if (!c)
        return;
    c->kind = Gull;
    c->anchor = {float(cam.x) + (rnd() - 0.5f) * 80.0f, float(cam.z) + (rnd() - 0.5f) * 80.0f};
    c->aux = rnd() * 2.0f * kPi;              // angle round the circle
    c->aux2 = 14.0f + rnd() * 22.0f;          // radius
    c->aux3 = (rnd() < 0.5f ? -1.0f : 1.0f) * (6.0f + rnd() * 3.0f) / c->aux2; // rad/s
    const double alt = std::max(cam.y, 66.0) + 20.0 + double(rnd()) * 26.0;
    c->pos = {double(c->anchor.x) + std::cos(c->aux) * c->aux2, alt,
              double(c->anchor.y) + std::sin(c->aux) * c->aux2};
    c->color = glm::vec3(0.96f, 0.97f, 1.0f);
    c->size = 0.9f + rnd() * 0.5f;
    c->phase = rnd() * 6.28f;
}

void Wildlife::spawnSchool(const ChunkManager& w, const glm::dvec3& cam, const Env& env) {
    int slot = -1;
    for (int i = 0; i < kSchools; ++i)
        if (!m_schools[i].alive) {
            slot = i;
            break;
        }
    if (slot < 0)
        return;
    const float ang = rnd() * 2.0f * kPi, r = 5.0f + rnd() * 24.0f;
    const int x = int(std::floor(cam.x + std::cos(ang) * r));
    const int z = int(std::floor(cam.z + std::sin(ang) * r));
    // walk down through air to the surface of the water, then on down to the bottom of it
    int surface = INT_MIN;
    for (int y = int(cam.y) + 12; y >= int(cam.y) - 34; --y) {
        const BlockId b = w.blockAtWorld({x, y, z});
        if (blocks::opaque(b))
            return; // hit ground before any water
        if (blocks::isWater(b)) {
            surface = y;
            break;
        }
    }
    if (surface == INT_MIN)
        return;
    const int floor = floorBelow(w, x, z, surface, 40);
    if (floor == INT_MIN || surface - floor < 3)
        return;
    School& s = m_schools[slot];
    s.alive = true;
    s.surface = float(surface) + 0.4f;
    s.floor = float(floor) + 1.0f;
    s.center = {double(x) + 0.5,
                double(s.floor + 1.2f + rnd() * std::max(s.surface - s.floor - 2.4f, 0.0f)),
                double(z) + 0.5};
    const float h = rnd() * 2.0f * kPi;
    s.vel = {std::cos(h) * 1.1f, 0.0f, std::sin(h) * 1.1f};

    const bool tropical = env.reef > 0.15f || rnd() < 0.35f;
    const glm::vec3 base = tropical ? kReefFish[int(rnd() * 4.999f)] : kOpenFish[int(rnd() * 3.999f)];
    const int n = 5 + int(rnd() * 5.0f);
    for (int i = 0; i < n; ++i) {
        Critter* c = acquire();
        if (!c)
            break;
        c->kind = Fish;
        c->group = slot;
        c->pos = s.center + glm::dvec3((rnd() - 0.5f) * 2.5f, (rnd() - 0.5f) * 1.2f,
                                       (rnd() - 0.5f) * 2.5f);
        c->vel = s.vel;
        c->color = base * (0.85f + rnd() * 0.3f);
        c->size = tropical ? 0.30f : 0.36f;
        c->phase = rnd() * 6.28f;
        c->aux = 0.8f + rnd() * 0.5f; // how eagerly it keeps up
    }
}

// Crabs live on sand at the edge of the water. Each keeps to its own bit of beach: it scuttles a
// few steps sideways, stops, turns, goes again, and turns back at the first step that would leave
// the sand or wet its feet.
void Wildlife::spawnCrab(const ChunkManager& w, const glm::dvec3& cam) {
    const float ang = rnd() * 2.0f * kPi, r = 4.0f + rnd() * 24.0f;
    const int x = int(std::floor(cam.x + std::cos(ang) * r));
    const int z = int(std::floor(cam.z + std::sin(ang) * r));
    const int y = floorBelow(w, x, z, int(cam.y) + 8, 20);
    if (y == INT_MIN)
        return;
    const BlockId ground = w.blockAtWorld({x, y, z});
    if ((ground != blocks::BeachSand && ground != blocks::Sand) || blocks::opaque(w.blockAtWorld({x, y + 1, z})) ||
        blocks::isWater(w.blockAtWorld({x, y + 1, z})))
        return;
    Critter* c = acquire();
    if (!c)
        return;
    c->kind = Crab;
    c->size = 0.62f + rnd() * 0.25f;
    c->pos = {double(x) + 0.5, double(y) + 1.0 + double(c->size) * 0.10, double(z) + 0.5};
    const float shade = 0.75f + rnd() * 0.35f;
    c->color = rnd() < 0.7f ? glm::vec3(0.85f, 0.30f, 0.16f) * shade : glm::vec3(0.72f, 0.55f, 0.34f) * shade;
    c->aux = rnd() * 6.28f;              // heading
    c->aux2 = float(y + 1);              // top of the sand under it
    c->aux3 = rnd() * 1.5f;              // time to the next stop or start
    c->vel = glm::vec3(std::cos(c->aux), 0.0f, std::sin(c->aux)) * 0.01f; // heading, at rest
    c->phase = rnd() * 6.28f;
}

bool Wildlife::findWater(const ChunkManager& w, const glm::dvec3& cam, float rMin, float rMax,
                         int minDepth, int& x, int& z, int& surface, int& floor) {
    const float ang = rnd() * 2.0f * kPi, r = rMin + rnd() * (rMax - rMin);
    x = int(std::floor(cam.x + std::cos(ang) * r));
    z = int(std::floor(cam.z + std::sin(ang) * r));
    surface = INT_MIN;
    for (int y = int(cam.y) + 12; y >= int(cam.y) - 34; --y) {
        const BlockId b = w.blockAtWorld({x, y, z});
        if (blocks::opaque(b))
            return false;
        if (blocks::isWater(b)) {
            surface = y;
            break;
        }
    }
    if (surface == INT_MIN)
        return false;
    floor = floorBelow(w, x, z, surface, 40);
    return floor != INT_MIN && surface - floor >= minDepth;
}

void Wildlife::spawnTurtle(const ChunkManager& w, const glm::dvec3& cam) {
    int x = 0, z = 0, surface = 0, floor = 0;
    if (!findWater(w, cam, 8.0f, 36.0f, 4, x, z, surface, floor))
        return;
    Critter* c = acquire();
    if (!c)
        return;
    c->kind = Turtle;
    c->size = 0.75f + rnd() * 0.25f;
    c->pos = {double(x) + 0.5, double(floor) + 1.4 + double(rnd()) * std::max(double(surface - floor) - 3.0, 0.5),
              double(z) + 0.5};
    c->aux = rnd() * 6.28f;
    c->color = glm::vec3(0.36f, 0.50f, 0.28f) * (0.8f + rnd() * 0.4f);
    c->phase = rnd() * 6.28f;
    c->vel = glm::vec3(std::cos(c->aux), 0.0f, std::sin(c->aux)) * 0.7f;
    c->aux2 = float(surface);
    c->aux3 = float(floor);
}

void Wildlife::spawnJelly(const ChunkManager& w, const glm::dvec3& cam) {
    int x = 0, z = 0, surface = 0, floor = 0;
    if (!findWater(w, cam, 4.0f, 30.0f, 4, x, z, surface, floor))
        return;
    Critter* c = acquire();
    if (!c)
        return;
    c->kind = Jelly;
    c->size = 0.5f + rnd() * 0.3f;
    c->pos = {double(x) + 0.5, double(floor) + 1.5 + double(rnd()) * std::max(double(surface - floor) - 3.5, 0.5),
              double(z) + 0.5};
    c->aux = rnd() * 6.28f;
    // bright enough (HDR) that the bloom takes them up after dark
    c->color = (rnd() < 0.5f ? glm::vec3(1.0f, 0.42f, 0.78f) : glm::vec3(0.40f, 0.85f, 1.0f)) * 1.5f;
    c->phase = rnd() * 6.28f;
    c->aux2 = float(surface);
    c->aux3 = float(floor);
}

void Wildlife::spawnLeaf(const ChunkManager& w, const glm::dvec3& cam) {
    const float ang = rnd() * 2.0f * kPi, r = 2.0f + rnd() * 13.0f;
    const int x = int(std::floor(cam.x + std::cos(ang) * r));
    const int z = int(std::floor(cam.z + std::sin(ang) * r));
    for (int y = int(cam.y) + 12; y >= int(cam.y) - 8; --y) {
        const BlockId b = w.blockAtWorld({x, y, z});
        if (blocks::isLeafBlock(b) &&
            !blocks::opaque(w.blockAtWorld({x, y - 1, z}))) {
            Critter* c = acquire();
            if (!c)
                return;
            c->kind = Leaf;
            c->pos = {double(x) + 0.2 + double(rnd()) * 0.6, double(y) - 0.1,
                      double(z) + 0.2 + double(rnd()) * 0.6};
            c->color = kLeafColours[rnd() < 0.85f ? int(rnd() * 2.999f) : 3];
            c->size = 0.07f + rnd() * 0.04f;
            c->phase = rnd() * 6.28f;
            c->aux = rnd() * 6.28f;
            c->life = 9.0f;
            return;
        }
        if (blocks::opaque(b))
            return; // ground before any canopy
    }
}

void Wildlife::spawnMote(const glm::dvec3& cam, const Env& env) {
    Critter* c = acquire();
    if (!c)
        return;
    // a point in a ball round the camera, biased to the front of the view is not needed: a
    // ball is fine, and the ones behind you are recycled forward as you turn
    glm::vec3 d;
    do {
        d = glm::vec3(rnd() * 2.0f - 1.0f, rnd() * 2.0f - 1.0f, rnd() * 2.0f - 1.0f);
    } while (glm::dot(d, d) > 1.0f);
    c->kind = Mote;
    c->pos = cam + glm::dvec3(d * 8.5f);
    c->color = env.underwater ? glm::vec3(0.62f, 0.86f, 0.92f) : glm::vec3(1.0f, 0.93f, 0.75f);
    c->size = 0.022f + rnd() * 0.02f;
    c->aux = rnd() * 6.28f;
    c->aux3 = env.underwater ? 0.4f : 0.55f; // alpha
}

// A small flock crossing the sky: they start well off to one side and fly a straight line
// that passes over the player, so they are seen, not just present. Each holds its place in a
// loose V.
void Wildlife::spawnFlock(const glm::dvec3& cam) {
    const float h = rnd() * 2.0f * kPi;
    const glm::vec3 dir(std::cos(h), 0.0f, std::sin(h));
    const glm::vec3 side(-dir.z, 0.0f, dir.x);
    const glm::dvec3 start = cam - glm::dvec3(dir * 95.0f) + glm::dvec3(side * ((rnd() - 0.5f) * 50.0f));
    const float alt = 20.0f + rnd() * 24.0f;
    const int n = 4 + int(rnd() * 3.0f);
    const float speed = 7.0f + rnd() * 3.0f;
    for (int i = 0; i < n; ++i) {
        Critter* c = acquire();
        if (!c)
            return;
        const float rank = float((i + 1) / 2) * (i % 2 == 0 ? 1.0f : -1.0f);
        c->kind = Bird;
        c->pos = start + glm::dvec3(dir * (-std::abs(rank) * 2.4f) + side * (rank * 2.2f)) +
                 glm::dvec3(0.0, cam.y + alt + (rnd() - 0.5f) * 1.5f, 0.0);
        c->vel = dir * speed;
        c->color = glm::vec3(0.16f, 0.15f, 0.17f);
        c->size = 0.55f + rnd() * 0.15f;
        c->phase = rnd() * 6.28f;
        c->aux = 0.9f + rnd() * 0.3f; // flap rate
        c->life = 26.0f;
    }
}

void Wildlife::spawnBubble(const ChunkManager& w, const glm::dvec3& cam) {
    const float ang = rnd() * 2.0f * kPi, r = 2.5f + rnd() * 9.0f;
    const int x = int(std::floor(cam.x + std::cos(ang) * r));
    const int z = int(std::floor(cam.z + std::sin(ang) * r));
    const int y = floorBelow(w, x, z, int(cam.y) + 3, 22);
    if (y == INT_MIN || !blocks::isWater(w.blockAtWorld({x, y + 1, z})))
        return;
    Critter* c = acquire();
    if (!c)
        return;
    c->kind = Bubble;
    c->pos = {double(x) + 0.3 + double(rnd()) * 0.4, double(y) + 1.1, double(z) + 0.3 + double(rnd()) * 0.4};
    c->vel = {0.0f, 1.1f + rnd() * 0.6f, 0.0f};
    c->color = glm::vec3(0.75f, 0.92f, 1.0f);
    c->size = 0.045f + rnd() * 0.03f;
    c->aux = rnd() * 6.28f;
    c->life = 14.0f;
}

void Wildlife::update(float dt, const ChunkManager& world, const glm::dvec3& cam, const Env& env) {
    dt = std::min(dt, 0.1f);
    m_time += dt;

    // ---- what the surroundings call for ------------------------------------------------
    const bool outdoors = !env.underwater && env.skyAccess > 0.55f;
    const bool day = env.daylight > 0.5f;
    int wantButterfly = 0, wantFirefly = 0, wantGull = 0, wantMote = 0;
    int wantCrab = 0, wantTurtle = 0, wantJelly = 0;
    if (outdoors && day && env.precip < 0.3f && env.desert < 0.5f && env.coast < 0.75f)
        wantButterfly = 8 + int(env.forest * 6.0f);
    if (outdoors && env.daylight < 0.22f && env.precip < 0.5f && env.desert < 0.6f)
        wantFirefly = 10 + int(22.0f * std::max(env.forest, env.nearWater));
    if (!env.underwater && env.coast > 0.3f && env.daylight > 0.3f && env.precip < 0.6f)
        wantGull = 4;
    if (env.underwater)
        wantMote = 55;
    else if (outdoors && env.daylight > 0.35f && env.precip < 0.5f)
        wantMote = 34;
    if (!env.underwater && env.skyAccess > 0.4f && env.coast > 0.35f && env.desert < 0.6f && env.precip < 0.5f)
        wantCrab = 3 + int(5.0f * env.coast);
    if (env.nearWater > 0.12f && (env.coast > 0.25f || env.reef > 0.1f) && env.precip < 0.7f)
        wantTurtle = env.underwater ? 3 : 2;
    // they show best at night, but are there by day too when you are down among them
    if ((env.daylight < 0.45f || env.underwater) && env.nearWater > 0.12f &&
        (env.reef > 0.12f || env.coast > 0.4f))
        wantJelly = env.underwater ? 12 : 8;
    int wantSchools = env.underwater ? 7 : (env.nearWater > 0.08f ? 4 : 0);
    if (env.reef > 0.2f)
        wantSchools += 2;
    wantSchools = std::min(wantSchools, kSchools);

    // ---- steer the schools, and let go of the ones the player has left ------------------
    int schools = 0;
    for (School& s : m_schools) {
        if (!s.alive)
            continue;
        const glm::dvec3 d = s.center - cam;
        if (glm::dot(d, d) > 52.0 * 52.0 || wantSchools == 0) {
            s.alive = false;
            continue;
        }
        ++schools;
        // a lazy random walk on heading, turned back the moment the next step would leave the water
        const float turn = (rnd() - 0.5f) * 1.6f * dt;
        const float cs = std::cos(turn), sn = std::sin(turn);
        s.vel = {s.vel.x * cs - s.vel.z * sn, 0.0f, s.vel.x * sn + s.vel.z * cs};
        const glm::dvec3 ahead = s.center + glm::dvec3(s.vel) * 1.2;
        if (!isWetAt(world, ahead) || isSolidAt(world, ahead))
            s.vel = {-s.vel.x + (rnd() - 0.5f) * 0.6f, 0.0f, -s.vel.z + (rnd() - 0.5f) * 0.6f};
        s.center += glm::dvec3(s.vel) * double(dt);
        const double bob = double(s.floor) + 1.6 + (double(s.surface - s.floor) - 3.2) * 0.5 *
                           (0.5 + 0.5 * std::sin(double(m_time) * 0.25 + double(&s - m_schools)));
        s.center.y += (bob - s.center.y) * double(std::min(dt * 0.5f, 1.0f));
    }

    // ---- per-creature behaviour --------------------------------------------------------
    for (int i = 0; i < m_live;) {
        Critter& c = m_pool[i];
        c.age += dt;
        c.life -= dt;
        const glm::dvec3 toCam = c.pos - cam;
        const double dist2 = glm::dot(toCam, toCam);
        bool dead = c.life <= 0.0f;

        // A shove or a punch rides on top of whatever the animal is doing and dies away. A crab
        // is thrown into a hop and falls back under gravity; everything else is in water or air
        // and just slows.
        if (c.flee > 0.0f)
            c.flee -= dt;
        if (glm::dot(c.kick, c.kick) > 1e-4f) {
            const bool hops = c.kind == Crab;
            if (hops)
                c.kick.y -= 16.0f * dt;
            const glm::dvec3 next = c.pos + glm::dvec3(c.kick) * double(dt);
            if (!isSolidAt(world, next))
                c.pos = next;
            else
                c.kick = glm::vec3(0.0f, std::min(c.kick.y, 0.0f), 0.0f);
            const float k = std::exp(-3.0f * dt);
            c.kick.x *= k;
            c.kick.z *= k;
            if (!hops)
                c.kick.y *= k;
        }

        switch (c.kind) {
            case Butterfly:
            case Firefly: {
                const bool bfly = c.kind == Butterfly;
                c.phase += dt * (bfly ? 19.0f : c.aux3);
                c.aux += (rnd() - 0.5f) * (bfly ? 9.0f : 3.0f) * dt;
                // startled: off the way it was sent, faster, and climbing
                const bool fleeing = bfly && c.flee > 0.0f;
                if (fleeing)
                    c.aux = std::atan2(c.fleeDir.z, c.fleeDir.x);
                const float spd = fleeing ? 3.2f
                                  : (bfly ? 1.3f + 0.5f * std::sin(c.age * 1.7f) : 0.35f + 0.15f * std::sin(c.age));
                c.vel.x = std::cos(c.aux) * spd;
                c.vel.z = std::sin(c.aux) * spd;
                // re-read the ground now and then: creatures follow the terrain, not a height
                if (std::fmod(c.age, 0.6f) < dt)
                    if (const int g = floorBelow(world, int(std::floor(c.pos.x)),
                                                 int(std::floor(c.pos.z)), int(c.pos.y) + 6, 14);
                        g != INT_MIN)
                        c.aux2 = float(g + 1);
                const float alt = c.aux2 + (fleeing ? 3.5f : 0.0f) +
                                  (bfly ? 1.1f + 0.8f * std::sin(c.age * 2.3f + c.aux)
                                        : 1.2f + 1.0f * std::sin(c.age * 0.7f + c.aux));
                c.vel.y = (alt - float(c.pos.y)) * 1.8f + std::sin(c.age * 7.0f) * (bfly ? 0.5f : 0.1f);
                const glm::dvec3 next = c.pos + glm::dvec3(c.vel) * double(dt);
                if (isSolidAt(world, next)) {
                    c.aux += 2.4f; // turn away from whatever that was
                    c.vel.y = 1.5f;
                } else {
                    c.pos = next;
                }
                if (dist2 > 44.0 * 44.0 || (bfly ? !day : env.daylight > 0.3f) || env.precip > 0.5f)
                    c.life = std::min(c.life, 0.6f);
                break;
            }
            case Gull: {
                c.aux += c.aux3 * dt;
                c.anchor.x += (float(cam.x) - c.anchor.x) * 0.04f * dt;
                c.anchor.y += (float(cam.z) - c.anchor.y) * 0.04f * dt;
                const float r = c.aux2;
                c.pos.x = double(c.anchor.x) + std::cos(c.aux) * r;
                c.pos.z = double(c.anchor.y) + std::sin(c.aux) * r;
                c.pos.y += std::sin(c.age * 0.6f + c.phase) * 0.6 * double(dt);
                c.vel = {-std::sin(c.aux) * r * c.aux3, 0.0f, std::cos(c.aux) * r * c.aux3};
                // beats now and then, glides between: the amplitude breathes
                c.phase += dt * (3.0f + 4.0f * (0.5f + 0.5f * std::sin(c.age * 0.35f)));
                if (wantGull == 0)
                    c.life = std::min(c.life, 0.6f);
                break;
            }
            case Bird: {
                c.pos += glm::dvec3(c.vel) * double(dt);
                c.pos.y += std::sin(c.age * 1.3f + c.aux) * 0.4 * double(dt);
                c.phase += dt * 15.0f * c.aux;
                if (dist2 > 120.0 * 120.0 || env.precip > 0.6f)
                    c.life = std::min(c.life, 0.6f);
                break;
            }
            case Fish: {
                if (c.group < 0 || !m_schools[c.group].alive) {
                    dead = true;
                    break;
                }
                const School& s = m_schools[c.group];
                const glm::vec3 pull = glm::vec3(s.center - c.pos);
                glm::vec3 want = s.vel * 1.0f + pull * 0.55f * c.aux;
                want += glm::vec3(std::sin(c.age * 1.9f + c.phase), std::sin(c.age * 1.3f) * 0.4f,
                                  std::cos(c.age * 1.7f + c.phase)) * 0.35f;
                // startled: a dart away, quicker than it ever swims otherwise, then back to the school
                const bool fleeing = c.flee > 0.0f;
                if (fleeing)
                    want = c.fleeDir * 4.6f + glm::vec3(0.0f, std::sin(c.phase) * 0.6f, 0.0f);
                c.vel += (want - c.vel) * std::min(1.0f, (fleeing ? 6.0f : 2.2f) * dt);
                const float sp = glm::length(c.vel);
                const float top = fleeing ? 5.0f : 2.2f;
                if (sp > top)
                    c.vel *= top / sp;
                const glm::dvec3 next = c.pos + glm::dvec3(c.vel) * double(dt);
                if (isWetAt(world, next) && !isSolidAt(world, next))
                    c.pos = next;
                else
                    c.vel = -c.vel * 0.5f;
                c.phase += dt * (8.0f + sp * 5.0f);
                break;
            }
            case Crab: {
                // startled: scuttling flat out, away, no pausing
                const bool fleeing = c.flee > 0.0f;
                if (fleeing) {
                    c.aux = std::atan2(c.fleeDir.z, c.fleeDir.x);
                    c.vel = glm::vec3(std::cos(c.aux), 0.0f, std::sin(c.aux)) * 3.0f;
                    c.aux3 = std::max(c.aux3, 0.4f);
                }
                // thrown or shoved somewhere new: find the ground under it again
                if (c.kick.x * c.kick.x + c.kick.z * c.kick.z > 0.01f)
                    if (const int g = floorBelow(world, int(std::floor(c.pos.x)), int(std::floor(c.pos.z)),
                                                 int(std::floor(c.pos.y)) + 1, 4);
                        g != INT_MIN)
                        c.aux2 = float(g + 1);
                const bool moving = glm::length(c.vel) > 0.05f;
                c.aux3 -= dt;
                if (c.aux3 <= 0.0f) {
                    if (moving) { // stop
                        c.vel = glm::vec3(std::cos(c.aux), 0.0f, std::sin(c.aux)) * 0.01f;
                        c.aux3 = 0.6f + 1.8f * rnd();
                    } else { // set off again, a little off the last heading
                        c.aux += (rnd() - 0.5f) * 2.4f;
                        c.vel = glm::vec3(std::cos(c.aux), 0.0f, std::sin(c.aux)) * 1.3f;
                        c.aux3 = 0.5f + 1.3f * rnd();
                    }
                }
                if (moving) {
                    const glm::dvec3 next = c.pos + glm::dvec3(c.vel) * double(dt);
                    const int nx = int(std::floor(next.x)), nz = int(std::floor(next.z));
                    const int g = floorBelow(world, nx, nz, int(std::floor(c.pos.y)) + 1, 3);
                    const BlockId under = g == INT_MIN ? blocks::Air : world.blockAtWorld({nx, g, nz});
                    const bool sand = under == blocks::BeachSand || under == blocks::Sand;
                    const bool clear = g != INT_MIN && !blocks::opaque(world.blockAtWorld({nx, g + 1, nz})) &&
                                       !blocks::isWater(world.blockAtWorld({nx, g + 1, nz}));
                    if (sand && clear) {
                        c.pos.x = next.x;
                        c.pos.z = next.z;
                        c.aux2 = float(g + 1);
                    } else if (fleeing) { // cornered at the edge of the sand: run along it instead
                        const float side = rnd() < 0.5f ? 1.0f : -1.0f;
                        c.fleeDir = glm::vec3(-c.fleeDir.z * side, 0.0f, c.fleeDir.x * side);
                    } else { // the edge of the sand: about face
                        c.aux += kPi + (rnd() - 0.5f) * 0.8f;
                        c.vel = glm::vec3(std::cos(c.aux), 0.0f, std::sin(c.aux)) * 1.3f;
                    }
                }
                const double rest = double(c.aux2) + double(c.size) * 0.10;
                if (c.kick.y != 0.0f) { // in the air off a punch: it lands under its own gravity
                    if (c.pos.y <= rest) {
                        c.pos.y = rest;
                        c.kick.y = 0.0f;
                    }
                } else {
                    c.pos.y += (rest - c.pos.y) * double(std::min(1.0f, dt * 12.0f));
                }
                c.phase += dt * (moving ? (fleeing ? 34.0f : 24.0f) : 1.5f);
                if (dist2 > 46.0 * 46.0 || wantCrab == 0)
                    c.life = std::min(c.life, 0.6f);
                break;
            }
            case Turtle:
            case Jelly: {
                const bool turtle = c.kind == Turtle;
                c.aux += (rnd() - 0.5f) * (turtle ? 0.9f : 1.6f) * dt;
                // startled: a turtle strokes away hard; a jellyfish can only drift off a little faster
                const bool fleeing = c.flee > 0.0f;
                if (fleeing)
                    c.aux = std::atan2(c.fleeDir.z, c.fleeDir.x);
                const float spd = turtle ? (fleeing ? 2.2f : 0.6f + 0.25f * std::sin(c.age * 0.5f))
                                         : (fleeing ? 0.6f : 0.22f);
                c.vel.x = std::cos(c.aux) * spd;
                c.vel.z = std::sin(c.aux) * spd;
                // turtles hold a depth and bob; jellyfish drift up and down through the water
                const float top = c.aux2 - 1.2f, bottom = c.aux3 + 1.2f;
                if (turtle)
                    c.vel.y = (0.5f * (top + bottom) + 0.6f * std::sin(c.age * 0.4f + c.phase) - float(c.pos.y)) * 0.6f;
                else
                    c.vel.y = std::sin(c.age * 0.7f + c.phase) * 0.45f;
                const glm::dvec3 next = c.pos + glm::dvec3(c.vel) * double(dt);
                if (isWetAt(world, next) && !isSolidAt(world, next)) {
                    c.pos = next;
                } else {
                    c.aux += 2.6f;
                    c.vel.y = 0.0f;
                }
                // a turtle beats its flippers in strokes with glides between; a jellyfish pulses
                c.phase += dt * (turtle ? 1.6f + 3.0f * std::max(0.0f, std::sin(c.age * 0.45f)) : 2.4f);
                if (dist2 > 52.0 * 52.0 || (turtle ? wantTurtle == 0 : wantJelly == 0))
                    c.life = std::min(c.life, 0.6f);
                break;
            }
            case Leaf: {
                c.phase += dt * 3.2f;
                c.vel.x = std::sin(c.age * 1.7f + c.aux) * 0.55f;
                c.vel.z = std::cos(c.age * 1.3f + c.aux) * 0.45f;
                c.vel.y = -0.85f - 0.3f * std::sin(c.age * 2.0f + c.aux);
                c.pos += glm::dvec3(c.vel) * double(dt);
                if (isSolidAt(world, c.pos) || isWetAt(world, c.pos))
                    dead = true;
                break;
            }
            case Mote: {
                c.vel += glm::vec3(std::sin(c.age * 0.9f + c.aux), std::sin(c.age * 0.7f + c.aux * 1.7f),
                                   std::cos(c.age * 0.8f + c.aux * 0.6f)) * 0.5f * dt;
                c.vel *= 1.0f - 0.9f * dt;
                c.pos += glm::dvec3(c.vel) * double(dt);
                if (dist2 > 11.0 * 11.0 || wantMote == 0)
                    dead = true;
                break;
            }
            case Bubble: {
                c.pos += glm::dvec3(c.vel) * double(dt);
                c.pos.x += std::sin(c.age * 3.0f + c.aux) * 0.25 * double(dt);
                c.pos.z += std::cos(c.age * 2.4f + c.aux) * 0.25 * double(dt);
                if (!isWetAt(world, c.pos) || dist2 > 30.0 * 30.0)
                    dead = true;
                break;
            }
        }

        if (dead) {
            m_pool[i] = m_pool[--m_live];
            continue;
        }
        ++i;
    }

    // ---- top up toward the targets, a couple of spawns a tick -----------------------------
    m_spawnClock += dt;
    if (m_spawnClock >= 0.1f) {
        m_spawnClock = 0.0f;
        if (count(Butterfly) < wantButterfly)
            spawnButterfly(world, cam);
        if (count(Firefly) < wantFirefly)
            spawnFirefly(world, cam);
        if (count(Gull) < wantGull)
            spawnGull(world, cam);
        if (schools < wantSchools)
            spawnSchool(world, cam, env);
        if (count(Crab) < wantCrab)
            spawnCrab(world, cam);
        if (count(Turtle) < wantTurtle)
            spawnTurtle(world, cam);
        if (count(Jelly) < wantJelly)
            spawnJelly(world, cam);
        for (int k = 0; k < 3; ++k)
            if (count(Mote) < wantMote)
                spawnMote(cam, env);
    }
    // a flock crosses now and then in fair weather; never two at once
    m_flockClock -= dt;
    if (m_flockClock <= 0.0f) {
        m_flockClock = 45.0f + rnd() * 70.0f;
        if (outdoors && day && env.precip < 0.4f && env.desert < 0.8f && count(Bird) == 0)
            spawnFlock(cam);
    }
    // leaves come off the canopy on their own clock, bubbles off the reef on another
    m_leafClock += dt;
    if (m_leafClock >= 0.3f) {
        m_leafClock = 0.0f;
        if (outdoors && env.forest > 0.12f && env.daylight > 0.15f && env.precip < 0.7f &&
            count(Leaf) < 24)
            spawnLeaf(world, cam);
    }
    m_bubbleClock += dt;
    if (m_bubbleClock >= (env.reef > 0.2f ? 0.16f : 0.7f)) {
        m_bubbleClock = 0.0f;
        if (env.underwater && count(Bubble) < 40)
            spawnBubble(world, cam);
    }
    m_light = 0.25f + 1.45f * env.daylight;
}

glm::vec2 Wildlife::bodyExtent(const Critter& c) {
    float r = 0.0f, hh = 0.0f;
    switch (c.kind) {
        case Butterfly: r = 0.6f, hh = 0.4f; break;
        case Gull:
        case Bird: r = 1.0f, hh = 0.25f; break;
        case Fish: r = 0.6f, hh = 0.3f; break;
        case Crab: r = 0.6f, hh = 0.25f; break;
        case Turtle: r = 0.7f, hh = 0.25f; break;
        case Jelly: r = 0.45f, hh = 0.5f; break;
        default: return glm::vec2(0.0f); // glows and leaves have nothing to bump into
    }
    // never smaller than something a player can reasonably catch with the crosshair
    return {std::max(r * c.size, 0.2f), std::max(hh * c.size, 0.15f)};
}

bool Wildlife::swimsAt(int index) const {
    const Kind k = m_pool[index].kind;
    return k == Fish || k == Turtle || k == Jelly;
}

void Wildlife::scare(Critter& c, const glm::vec3& away, float seconds) {
    glm::vec3 flat(away.x, 0.0f, away.z);
    const float len = glm::length(flat);
    flat = len > 1e-4f ? flat / len : glm::vec3(1.0f, 0.0f, 0.0f);
    c.flee = std::max(c.flee, seconds);
    c.fleeDir = flat;
    // a fish on its own would only swim back into the school it bolted from: move the school
    if (c.kind == Fish && c.group >= 0 && m_schools[c.group].alive)
        m_schools[c.group].vel = flat * 1.8f;
}

glm::dvec2 Wildlife::interact(const Body& p, const ChunkManager& world) {
    glm::dvec2 give(0.0);
    const glm::vec2 pv(p.vel.x, p.vel.z);
    const double pr = double(p.halfWidth) * 1.15; // a little round the box's corners
    for (int i = 0; i < m_live; ++i) {
        Critter& c = m_pool[i];
        const glm::vec2 ext = bodyExtent(c);
        if (ext.x <= 0.0f)
            continue;
        const glm::dvec2 d(c.pos.x - p.feet.x, c.pos.z - p.feet.z);
        const double dist = glm::length(d);
        if (dist > 6.0)
            continue;
        const glm::vec3 away = dist > 1e-4 ? glm::vec3(float(d.x / dist), 0.0f, float(d.y / dist))
                                           : glm::vec3(1.0f, 0.0f, 0.0f);
        const bool level = c.pos.y + double(ext.y) > p.feet.y && c.pos.y - double(ext.y) < p.feet.y + double(p.height);
        // Crowded: walked straight at, or swum in among. They go before they are touched.
        const bool swimmer = c.kind == Fish || c.kind == Turtle || c.kind == Jelly;
        const float closing = glm::dot(pv, glm::vec2(away.x, away.z));
        if (level && ((dist < 3.5 && closing > 1.5f) || (swimmer && dist < 2.2)))
            scare(c, away, 2.5f);
        // Touching: pushed apart. The animal is the lighter one, so it takes most of it, and if
        // there is a wall behind it the player takes all of it instead of shoving it into the wall.
        const double rsum = pr + double(ext.x);
        if (!level || dist >= rsum)
            continue;
        scare(c, away, 3.0f);
        if (!p.solid)
            continue;
        const double pen = rsum - dist;
        const double share = c.kind == Turtle ? 0.65 : 0.9;
        const glm::dvec3 to = c.pos + glm::dvec3(away) * (pen * share);
        if (!isSolidAt(world, to)) {
            c.pos = to;
            c.kick += away * 1.2f;
            give -= glm::dvec2(away.x, away.z) * (pen * (1.0 - share));
        } else {
            give -= glm::dvec2(away.x, away.z) * pen;
        }
    }
    return give;
}

int Wildlife::pick(const glm::dvec3& origin, const glm::vec3& dir, float maxDist, float& dist) const {
    int best = -1;
    float bestT = maxDist;
    for (int i = 0; i < m_live; ++i) {
        const Critter& c = m_pool[i];
        const glm::vec2 ext = bodyExtent(c);
        if (ext.x <= 0.0f)
            continue;
        const glm::vec3 rel(c.pos - origin);
        const glm::vec3 lo = rel - glm::vec3(ext.x, ext.y, ext.x);
        const glm::vec3 hi = rel + glm::vec3(ext.x, ext.y, ext.x);
        float t0 = 0.0f, t1 = bestT;
        bool hit = true;
        for (int a = 0; a < 3 && hit; ++a) {
            if (std::abs(dir[a]) < 1e-6f) {
                hit = lo[a] <= 0.0f && hi[a] >= 0.0f;
                continue;
            }
            float ta = lo[a] / dir[a], tb = hi[a] / dir[a];
            if (ta > tb)
                std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
            hit = t0 <= t1;
        }
        if (hit) {
            bestT = t0;
            best = i;
        }
    }
    dist = bestT;
    return best;
}

void Wildlife::punch(int index, const glm::vec3& dir) {
    Critter& c = m_pool[index];
    glm::vec3 flat(dir.x, 0.0f, dir.z);
    const float len = glm::length(flat);
    flat = len > 1e-4f ? flat / len : glm::vec3(1.0f, 0.0f, 0.0f);
    if (c.kind == Crab)
        c.kick = flat * 5.5f + glm::vec3(0.0f, 4.5f, 0.0f); // knocked off its feet
    else if (swimsAt(index))
        c.kick = glm::normalize(dir) * (c.kind == Turtle ? 3.2f : 5.0f);
    else
        c.kick = flat * 4.0f + glm::vec3(0.0f, 2.0f, 0.0f);
    scare(c, flat, 4.0f);
}

void Wildlife::draw(const glm::mat4& viewProj, const glm::dvec3& cam, const glm::vec3& right,
                    const glm::vec3& up, bool underwater) {
    if (m_live == 0 || m_prog == 0 || m_prog3d == 0)
        return;
    // Two batches packed into one buffer, in draw order: the glows (sprites, blended, no depth
    // write), then the animals (solid boxes, depth write).
    const auto batchOf = [](Kind k) { return isSolid(k) ? 1 : 0; };
    auto* map = static_cast<GpuSprite*>(m_ring.begin());
    int out = 0;
    int batchEnd[2] = {0, 0};
    for (int batch = 0; batch < 2; ++batch) {
        for (int i = 0; i < m_live; ++i) {
            const Critter& c = m_pool[i];
            if (batchOf(c.kind) != batch)
                continue;
            const glm::vec3 rel(c.pos - cam);
            const float d = glm::length(rel);
            const float far = underwater ? 26.0f : (c.kind == Gull ? 140.0f : 92.0f);
            float a = std::min(1.0f, c.age * 2.0f) * std::clamp(c.life / 0.6f, 0.0f, 1.0f) *
                      (1.0f - glm::smoothstep(far * 0.6f, far, d));
            const bool emissive = c.kind == Firefly || c.kind == Jelly;
            const float lit = emissive ? 1.0f : m_light;
            // Sprites turn to a heading in the camera's own plane; bodies need the real one, along
            // the ground.
            glm::vec2 h;
            if (isSolid(c.kind))
                h = {c.vel.x, c.vel.z};
            else
                h = {glm::dot(c.vel, right), glm::dot(c.vel, up)};
            if (glm::dot(h, h) < 1e-6f)
                h = {1.0f, 0.0f};
            if (c.kind == Mote)
                a *= c.aux3;
            GpuSprite& g = map[out++];
            g.x = rel.x;
            g.y = rel.y;
            g.z = rel.z;
            g.size = c.size;
            g.r = c.color.r * lit;
            g.g = c.color.g * lit;
            g.b = c.color.b * lit;
            g.a = a;
            g.kind = float(c.kind);
            g.dirX = h.x;
            g.dirY = h.y;
            g.phase = c.phase;
        }
        batchEnd[batch] = out;
    }
    // 36 vertices a box, eight boxes a creature; keep in step with kParts in wildlife3d.vert
    constexpr int kVertsPerBody = 8 * 36;

    glUseProgram(m_prog);
    glProgramUniformMatrix4fv(m_prog, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
    glProgramUniform3fv(m_prog, 4, 1, &right.x);
    glProgramUniform3fv(m_prog, 5, 1, &up.x);
    m_ring.bind(0);
    glBindVertexArray(m_vao);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    if (batchEnd[0] > 0)
        glDrawArrays(GL_TRIANGLES, 0, batchEnd[0] * 6);
    glDisable(GL_BLEND);

    if (out > batchEnd[0]) {
        glUseProgram(m_prog3d);
        glProgramUniformMatrix4fv(m_prog3d, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glDepthMask(GL_TRUE);
        glDrawArrays(GL_TRIANGLES, batchEnd[0] * kVertsPerBody, (out - batchEnd[0]) * kVertsPerBody);
    }
    glDepthMask(GL_TRUE);
    m_ring.fence();
}

} // namespace vx
