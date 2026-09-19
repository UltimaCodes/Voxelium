#include "render/Wildlife.hpp"

#include "core/Log.hpp"
#include "gfx/Shader.hpp"
#include "world/ChunkManager.hpp"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <climits>
#include <cmath>

namespace vx {

namespace {

constexpr GLbitfield kWriteMapFlags =
    GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
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
const glm::vec3 kOpenFish[] = {{0.62f, 0.72f, 0.82f}, {0.50f, 0.62f, 0.34f},
                               {0.85f, 0.56f, 0.42f}, {0.45f, 0.55f, 0.75f}};
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
    glCreateBuffers(1, &m_ssbo);
    const GLsizeiptr bytes = GLsizeiptr(kCapacity) * GLsizeiptr(sizeof(GpuSprite));
    glNamedBufferStorage(m_ssbo, bytes, nullptr, kWriteMapFlags);
    m_map = static_cast<GpuSprite*>(glMapNamedBufferRange(m_ssbo, 0, bytes, kWriteMapFlags));
    if (!m_map) {
        VX_ERROR("wildlife: persistent mapping failed");
        return false;
    }
    return true;
}

void Wildlife::shutdown() {
    if (m_prog)
        glDeleteProgram(m_prog);
    if (m_prog3d)
        glDeleteProgram(m_prog3d);
    m_prog = m_prog3d = 0;
    if (m_ssbo)
        glDeleteBuffers(1, &m_ssbo);
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    m_ssbo = m_vao = 0;
    m_map = nullptr;
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

        switch (c.kind) {
            case Butterfly:
            case Firefly: {
                const bool bfly = c.kind == Butterfly;
                c.phase += dt * (bfly ? 19.0f : c.aux3);
                c.aux += (rnd() - 0.5f) * (bfly ? 9.0f : 3.0f) * dt;
                const float spd = bfly ? 1.3f + 0.5f * std::sin(c.age * 1.7f) : 0.35f + 0.15f * std::sin(c.age);
                c.vel.x = std::cos(c.aux) * spd;
                c.vel.z = std::sin(c.aux) * spd;
                // re-read the ground now and then: creatures follow the terrain, not a height
                if (std::fmod(c.age, 0.6f) < dt)
                    if (const int g = floorBelow(world, int(std::floor(c.pos.x)),
                                                 int(std::floor(c.pos.z)), int(c.pos.y) + 6, 14);
                        g != INT_MIN)
                        c.aux2 = float(g + 1);
                const float alt = c.aux2 + (bfly ? 1.1f + 0.8f * std::sin(c.age * 2.3f + c.aux)
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
                c.vel += (want - c.vel) * std::min(1.0f, 2.2f * dt);
                const float sp = glm::length(c.vel);
                if (sp > 2.2f)
                    c.vel *= 2.2f / sp;
                const glm::dvec3 next = c.pos + glm::dvec3(c.vel) * double(dt);
                if (isWetAt(world, next) && !isSolidAt(world, next))
                    c.pos = next;
                else
                    c.vel = -c.vel * 0.5f;
                c.phase += dt * (8.0f + sp * 5.0f);
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

void Wildlife::placeDemo(const glm::dvec3& cam, const glm::vec3& forward) {
    const glm::vec3 fwd = glm::normalize(glm::vec3(forward.x, 0.0f, forward.z));
    const glm::vec3 side(-fwd.z, 0.0f, fwd.x);
    struct D { Kind k; float size; glm::vec3 col; float heading; };
    const D demo[] = {{Butterfly, 0.6f, {0.95f, 0.55f, 0.15f}, 0.3f}, {Gull, 0.9f, {0.95f, 0.95f, 0.95f}, 1.2f},
                      {Bird, 0.7f, {0.16f, 0.16f, 0.18f}, 2.0f},      {Fish, 0.8f, {1.0f, 0.5f, 0.1f}, 0.0f},
                      {Leaf, 0.5f, {0.8f, 0.45f, 0.1f}, 0.0f}};
    int i = 0;
    for (const D& d : demo) {
        Critter* c = acquire();
        if (!c)
            return;
        c->kind = d.k;
        c->size = d.size;
        c->color = d.col;
        c->pos = cam + glm::dvec3(fwd * 5.0f + side * (float(i) - 2.0f) * 2.2f);
        c->vel = glm::vec3(std::cos(d.heading), 0.0f, std::sin(d.heading)) * 0.5f;
        c->age = 5.0f;
        c->life = 1e9f;
        c->phase = 0.4f * float(i);
        ++i;
    }
}

void Wildlife::draw(const glm::mat4& viewProj, const glm::dvec3& cam, const glm::vec3& right,
                    const glm::vec3& up, bool underwater) {
    if (m_live == 0 || m_prog == 0 || m_prog3d == 0)
        return;
    // Three batches packed into one buffer, in draw order: the glows (sprites, blended, no depth
    // write), then the fish (solid, no depth write), then the airborne animals (solid, depth
    // write).
    const auto batchOf = [](Kind k) { return !isSolid(k) ? 0 : (writesDepth(k) ? 2 : 1); };
    int out = 0;
    int batchEnd[3] = {0, 0, 0};
    for (int batch = 0; batch < 3; ++batch) {
        for (int i = 0; i < m_live; ++i) {
            const Critter& c = m_pool[i];
            if (batchOf(c.kind) != batch)
                continue;
            const glm::vec3 rel(c.pos - cam);
            const float d = glm::length(rel);
            const float far = underwater ? 26.0f : (c.kind == Gull ? 140.0f : 92.0f);
            float a = std::min(1.0f, c.age * 2.0f) * std::clamp(c.life / 0.6f, 0.0f, 1.0f) *
                      (1.0f - glm::smoothstep(far * 0.6f, far, d));
            const bool emissive = c.kind == Firefly;
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
            GpuSprite& g = m_map[out++];
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
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo);
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
        if (batchEnd[1] > batchEnd[0])
            glDrawArrays(GL_TRIANGLES, batchEnd[0] * kVertsPerBody,
                         (batchEnd[1] - batchEnd[0]) * kVertsPerBody);
        glDepthMask(GL_TRUE);
        if (batchEnd[2] > batchEnd[1])
            glDrawArrays(GL_TRIANGLES, batchEnd[1] * kVertsPerBody,
                         (batchEnd[2] - batchEnd[1]) * kVertsPerBody);
    }
    glDepthMask(GL_TRUE);
}

} // namespace vx
