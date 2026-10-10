#include "render/Particles.hpp"

#include "core/Log.hpp"
#include "gfx/BlockTextures.hpp"
#include "gfx/Shader.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkManager.hpp"

#include <glad/gl.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>

namespace vx {

namespace {
constexpr float kGravity = 14.0f; // lighter than the player's 32: debris should hang a bit

// The same gust wave chunk.vert bends the grass with, so sand streams across the ground in the
// gusts that are also leaning the plants: slow, travelling in world space, mostly a breeze.
float gust(double x, double z, float t) {
    const float g = 0.5f + 0.5f * std::sin(float(x * 0.047 + z * 0.031) + t * 0.45f);
    return 0.55f + 1.2f * g * g;
}
const glm::vec3 kWindDir = glm::normalize(glm::vec3(0.83f, 0.0f, 0.55f));

// How much a block puts up when stepped on or broken: loose stuff clouds, hard stuff chips.
int puffFor(int id) {
    switch (id) {
        case blocks::Sand:
        case blocks::BeachSand: return 10;
        case blocks::Snow: return 8;
        case blocks::Gravel:
        case blocks::Dirt: return 5;
        case blocks::Sandstone: return 3;
        case blocks::Grass: return 2;
        default: return 0;
    }
}
} // namespace

float Particles::m_rand01() {
    // xorshift rather than <random>: this is called a few thousand times per burst and
    // must not allocate or lock
    m_rng ^= m_rng << 13;
    m_rng ^= m_rng >> 17;
    m_rng ^= m_rng << 5;
    return float(m_rng & 0xFFFFFFu) / float(0x1000000u);
}

bool Particles::init(const std::filesystem::path& shaderDir) {
    m_prog = loadProgram(shaderDir / "particle.vert", shaderDir / "particle.frag");
    if (!m_prog)
        return false;
    glCreateVertexArrays(1, &m_vao);
    return m_ring.init(std::size_t(kCapacity) * sizeof(GpuParticle));
}

void Particles::shutdown() {
    m_ring.shutdown();
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    m_vao = 0;
    m_live = 0;
}

Particles::Particle& Particles::acquire() {
    if (m_live < kCapacity)
        return m_pool[m_live++];
    // Full: overwrite in turn. It used to be slot 0 every time, so a whole burst landed in one
    // slot and a storm (a bolt is a couple of thousand particles) left one shard of every
    // block broken during it. No slot is reliably the oldest, compaction swaps the newest into
    // each freed slot, so this only spreads the damage, which is all a full pool can do.
    m_victim = (m_victim + 1) % kCapacity;
    return m_pool[m_victim];
}

int Particles::countLayer(int layer) const {
    int n = 0;
    for (int i = 0; i < m_live; ++i)
        n += m_pool[i].layer == layer ? 1 : 0;
    return n;
}

Particles::Particle& Particles::fresh() {
    Particle& p = acquire();
    p = Particle{};
    return p;
}

// A cloud thrown up off loose ground: soft, wide, and slow to fall, tinted by the ground's own
// texture so a step in sand throws sand and a step in snow throws snow.
void Particles::puff(const glm::dvec3& at, int texLayer, int count, float spread, float lift) {
    for (int i = 0; i < count; ++i) {
        Particle& p = fresh();
        const float a = m_rand01() * glm::two_pi<float>();
        const float r = m_rand01();
        p.pos = at + glm::dvec3(std::cos(a) * r * 0.35, 0.06 + m_rand01() * 0.12,
                                std::sin(a) * r * 0.35);
        p.vel = glm::vec3(std::cos(a), 0.0f, std::sin(a)) * spread * (0.4f + 0.6f * m_rand01()) +
                glm::vec3(0.0f, lift * (0.5f + m_rand01()), 0.0f);
        p.life = 0.7f + m_rand01() * 0.7f;
        p.invLife = 1.0f / p.life;
        p.size = 0.07f + m_rand01() * 0.07f;
        p.growth = 0.16f + m_rand01() * 0.12f;
        p.layer = texLayer;
        p.gravity = 0.10f;
        p.drag = 2.4f;
        p.maxAlpha = 0.55f;
        p.wind = 0.6f;
    }
}

void Particles::spawnBlockBreak(const glm::dvec3& blockMin, int texLayer, int blockId) {
    // Chunks of the block itself: they bounce off the ground, slide, and lie there a moment.
    for (int i = 0; i < 14; ++i) {
        Particle& p = fresh();
        p.pos = blockMin + glm::dvec3(m_rand01(), m_rand01(), m_rand01());
        p.vel = glm::vec3(m_rand01() - 0.5f, m_rand01() * 0.9f + 0.15f, m_rand01() - 0.5f) *
                4.2f;
        p.life = 1.1f + m_rand01() * 0.9f;
        p.invLife = 1.0f / p.life;
        p.size = 0.055f + m_rand01() * 0.055f;
        p.layer = texLayer;
        p.gravity = 1.0f;
        p.drag = 0.15f;
        p.bounce = 0.38f;
        p.collide = true;
    }
    if (puffFor(blockId) > 0)
        puff(blockMin + glm::dvec3(0.5, 0.2, 0.5), texLayer, puffFor(blockId) * 2, 1.6f, 0.9f);
}

void Particles::spawnSplash(const glm::dvec3& at) {
    for (int i = 0; i < 10; ++i) {
        Particle& p = fresh();
        p.pos = at + glm::dvec3(m_rand01() * 0.6 - 0.3, 0.05, m_rand01() * 0.6 - 0.3);
        // biased upward and outward: a splash is a crown, not a puff
        p.vel = glm::vec3((m_rand01() - 0.5f) * 2.6f, 2.6f + m_rand01() * 2.4f,
                          (m_rand01() - 0.5f) * 2.6f);
        p.life = 0.35f + m_rand01() * 0.3f;
        p.invLife = 1.0f / p.life;
        p.size = 0.05f + m_rand01() * 0.04f;
        p.layer = -1; // flat water colour; no block texture reads as a droplet
    }
}

void Particles::spawnFootstep(const glm::dvec3& at, int texLayer, int blockId) {
    for (int i = 0; i < 4; ++i) {
        Particle& p = fresh();
        p.pos = at + glm::dvec3(m_rand01() * 0.4 - 0.2, 0.04, m_rand01() * 0.4 - 0.2);
        p.vel = glm::vec3((m_rand01() - 0.5f) * 1.1f, m_rand01() * 0.7f,
                          (m_rand01() - 0.5f) * 1.1f);
        p.life = 0.5f + m_rand01() * 0.3f;
        p.invLife = 1.0f / p.life;
        p.size = 0.05f + m_rand01() * 0.04f;
        p.layer = texLayer;
        p.bounce = 0.3f;
        p.drag = 0.4f;
        p.collide = true;
    }
    if (const int n = puffFor(blockId); n > 0)
        puff(at, texLayer, n, 0.9f, 0.45f);
}

void Particles::spawnLanding(const glm::dvec3& at, int texLayer, int blockId) {
    // a ring thrown outward along the ground, and a lot more of it
    const int n = std::max(puffFor(blockId), 4) * 3;
    puff(at, texLayer, n, 2.4f, 0.7f);
    for (int i = 0; i < 6; ++i) {
        Particle& p = fresh();
        p.pos = at + glm::dvec3(m_rand01() * 0.6 - 0.3, 0.05, m_rand01() * 0.6 - 0.3);
        p.vel = glm::vec3((m_rand01() - 0.5f) * 3.6f, 1.2f + m_rand01() * 1.6f,
                          (m_rand01() - 0.5f) * 3.6f);
        p.life = 0.8f + m_rand01() * 0.6f;
        p.invLife = 1.0f / p.life;
        p.size = 0.05f + m_rand01() * 0.04f;
        p.layer = texLayer;
        p.bounce = 0.3f;
        p.collide = true;
    }
}

void Particles::spawnBolt(const glm::dvec3& top, const glm::dvec3& bottom, std::uint32_t seed,
                          float thickness) {
    std::uint32_t rs = seed ^ 0x9E3779B9u;
    const auto r01 = [&rs]() {
        rs = rs * 1664525u + 1013904223u;
        return float(rs >> 8) / float(1 << 24);
    };
    // one polyline by midpoint displacement: each pass halves the segments and knocks every new
    // midpoint sideways by less than the last, which is what gives a bolt its kinks at every scale
    const auto jagged = [&](const glm::dvec3& a, const glm::dvec3& b, glm::dvec3* pts, int segs,
                            double amp) {
        pts[0] = a;
        pts[segs] = b;
        for (int step = segs / 2; step >= 1; step /= 2) {
            for (int i = step; i < segs; i += 2 * step) {
                glm::dvec3 mid = (pts[i - step] + pts[i + step]) * 0.5;
                mid.x += double(r01() - 0.5f) * 2.0 * amp;
                mid.z += double(r01() - 0.5f) * 2.0 * amp;
                pts[i] = mid;
            }
            amp *= 0.55;
        }
    };
    const auto lay = [&](const glm::dvec3* pts, int segs, float width, float bright) {
        const float spacing = std::max(0.35f, width * 0.7f);
        for (int i = 0; i < segs; ++i) {
            const glm::dvec3 d = pts[i + 1] - pts[i];
            const int n = std::max(1, int(float(glm::length(d)) / spacing));
            for (int k = 0; k < n; ++k) {
                Particle& p = fresh();
                p.pos = pts[i] + d * (double(k) / double(n));
                p.vel = glm::vec3(0.0f);
                p.life = 0.20f + 0.08f * r01();
                p.invLife = 1.0f / p.life;
                p.size = width * (0.8f + 0.4f * r01());
                p.layer = -1;
                p.gravity = 0.0f;
                p.tint = glm::vec3(4.5f, 5.2f, 7.6f) * bright * (0.8f + 0.4f * r01());
                p.maxAlpha = 1.0f;
                p.solid = true; // a bolt is mostly against the sky, which the composite repaints
            }
        }
    };

    constexpr int kSegs = 64;
    glm::dvec3 pts[kSegs + 1];
    const double len = glm::length(top - bottom);
    jagged(top, bottom, pts, kSegs, len * 0.10);
    lay(pts, kSegs, thickness, 1.0f);

    // forks: shorter, thinner, dimmer lines leaving the main one on the way down
    for (int f = 0; f < 3; ++f) {
        const int at = 10 + int(r01() * 38.0f);
        const glm::dvec3 from = pts[at];
        const glm::dvec3 to = from + glm::dvec3((r01() - 0.5f) * len * 0.35, -len * 0.22 * (0.5 + r01()),
                                                (r01() - 0.5f) * len * 0.35);
        glm::dvec3 branch[17];
        jagged(from, to, branch, 16, len * 0.04);
        lay(branch, 16, thickness * 0.55f, 0.55f);
    }
}

// Highest solid or water cell in the column between two heights, top down.
bool Particles::groundBelow(const ChunkManager& w, int x, int z, int fromY, int toY, int& outY,
                            int& outBlock) const {
    for (int y = fromY; y >= toY; --y) {
        const BlockId id = w.blockAtWorld(glm::ivec3(x, y, z));
        if (id != blocks::Air && (blocks::opaque(id) || blocks::isWater(id))) {
            outY = y;
            outBlock = int(id);
            return true;
        }
    }
    return false;
}

void Particles::emitAmbient(float dt, const ChunkManager& world, const glm::dvec3& cam,
                            const Ambient& a) {
    m_time = a.time;
    if (a.underwater || dt <= 0.0f)
        return;
    const int cx = int(std::floor(cam.x)), cz = int(std::floor(cam.z));
    const int cy = int(std::floor(cam.y));

    // --- wind-blown sand: streams of grains low over sand, stronger in the gusts
    const float sandy = a.desert + 0.25f * a.coast;
    if (sandy > 0.02f) {
        const float g = gust(cam.x, cam.z, a.time);
        m_sandCarry += sandy * 110.0f * g * dt;
        int tries = 0;
        while (m_sandCarry >= 1.0f && tries++ < 12) {
            m_sandCarry -= 1.0f;
            const float ang = m_rand01() * glm::two_pi<float>();
            const float rad = 3.0f + m_rand01() * 20.0f;
            const int x = cx + int(std::lround(std::cos(ang) * rad));
            const int z = cz + int(std::lround(std::sin(ang) * rad));
            int gy = 0, gb = 0;
            if (!groundBelow(world, x, z, cy + 10, cy - 18, gy, gb))
                continue;
            if (gb != blocks::Sand && gb != blocks::BeachSand)
                continue;
            Particle& p = fresh();
            p.pos = glm::dvec3(double(x) + m_rand01(), double(gy) + 1.05, double(z) + m_rand01());
            p.vel = kWindDir * (2.5f + 3.5f * g * m_rand01()) + glm::vec3(0.0f, 0.3f + m_rand01() * 1.1f, 0.0f);
            p.life = 1.4f + m_rand01() * 1.8f;
            p.invLife = 1.0f / p.life;
            p.size = 0.09f + m_rand01() * 0.07f;
            p.layer = blockTopLayer(gb);
            p.gravity = 0.14f;
            p.drag = 0.5f;
            p.bounce = 0.25f;
            p.maxAlpha = 0.9f;
            p.wind = 1.0f;
            p.collide = true;
        }
    }

    // --- rain splashes: a small crown of droplets where rain lands, near enough to see
    if (a.precip > 0.02f && a.snow < 0.5f) {
        m_rainCarry += a.precip * 75.0f * dt;
        int tries = 0;
        while (m_rainCarry >= 1.0f && tries++ < 20) {
            m_rainCarry -= 1.0f;
            const float ang = m_rand01() * glm::two_pi<float>();
            const float rad = 1.5f + std::sqrt(m_rand01()) * 15.0f;
            const int x = cx + int(std::lround(std::cos(ang) * rad));
            const int z = cz + int(std::lround(std::sin(ang) * rad));
            int gy = 0, gb = 0;
            // from above the camera: the first thing rain meets is a roof or a leaf, not the
            // floor under it, which is what keeps splashes off the ground inside a house
            if (!groundBelow(world, x, z, cy + 16, cy - 24, gy, gb))
                continue;
            const bool onWater = blocks::isWater(BlockId(gb));
            const double top = onWater ? double(gy) + 0.85 : double(gy) + 1.0;
            for (int k = 0; k < 3; ++k) {
                Particle& p = fresh();
                p.pos = glm::dvec3(double(x) + m_rand01(), top + 0.02, double(z) + m_rand01());
                p.vel = glm::vec3((m_rand01() - 0.5f) * 1.6f, 1.4f + m_rand01() * 1.8f,
                                  (m_rand01() - 0.5f) * 1.6f);
                p.life = 0.22f + m_rand01() * 0.16f;
                p.invLife = 1.0f / p.life;
                p.size = 0.016f + m_rand01() * 0.014f;
                p.layer = -1;
                p.maxAlpha = 0.85f;
            }
        }
    }
}

void Particles::update(float dt, const ChunkManager* world) {
    // Swap-remove into a compacted prefix: the draw call then submits [0, m_live) with no
    // gaps and no per-particle liveness test in the shader.
    for (int i = 0; i < m_live;) {
        Particle& p = m_pool[i];
        p.life -= dt;
        if (p.life <= 0.0f) {
            m_pool[i] = m_pool[--m_live];
            continue;
        }
        ++i;
        if (p.resting)
            continue;

        p.vel.y -= kGravity * p.gravity * dt;
        if (p.drag > 0.0f)
            p.vel *= std::max(0.0f, 1.0f - p.drag * dt);
        if (p.wind > 0.0f) {
            // ease toward the local wind rather than being shoved by it, so a grain accelerates
            // into a gust and hangs on after it
            const float gs = gust(p.pos.x, p.pos.z, m_time);
            const glm::vec3 target = kWindDir * (3.2f * gs);
            const float k = std::min(1.0f, 1.6f * dt) * p.wind;
            p.vel.x += (target.x - p.vel.x) * k;
            p.vel.z += (target.z - p.vel.z) * k;
        }
        p.size += p.growth * dt;

        const glm::dvec3 next = p.pos + glm::dvec3(p.vel) * double(dt);
        if (world == nullptr || !p.collide) {
            p.pos = next;
            continue;
        }
        const auto solidAt = [&](const glm::dvec3& q) {
            return blocks::opaque(world->blockAtWorld(glm::ivec3(glm::floor(q))));
        };
        // Spawned inside the ground (a shard from a block that has just gone, a grain on a
        // slope): lift out instead of sticking there for its whole life.
        if (solidAt(p.pos))
            p.pos.y = std::floor(p.pos.y) + 1.02;
        glm::dvec3 res = p.pos;
        // one axis at a time, so a grain sliding along the ground keeps its sideways speed
        if (!solidAt({next.x, res.y, res.z}))
            res.x = next.x;
        else
            p.vel.x *= -p.bounce;
        if (!solidAt({res.x, res.y, next.z}))
            res.z = next.z;
        else
            p.vel.z *= -p.bounce;
        if (!solidAt({res.x, next.y, res.z})) {
            res.y = next.y;
        } else {
            const bool ground = p.vel.y < 0.0f;
            p.vel.y *= -p.bounce;
            p.vel.x *= 0.7f; // friction: scuffing along the ground
            p.vel.z *= 0.7f;
            if (ground && std::abs(p.vel.y) < 0.7f) {
                p.vel = glm::vec3(0.0f);
                p.resting = true; // lies there until its life runs out
            }
        }
        p.pos = res;
        if (blocks::isWater(world->blockAtWorld(glm::ivec3(glm::floor(p.pos)))))
            p.vel *= std::max(0.0f, 1.0f - 6.0f * dt); // water stops things dead
    }
}

void Particles::draw(const glm::mat4& viewProj, const glm::dvec3& cameraPos,
                     const glm::vec3& right, const glm::vec3& up,
                     unsigned int blockTexArray) {
    if (m_live == 0)
        return;
    // Two batches in one buffer: the ordinary ones (blended, no depth write) first, then the ones
    // that must write depth so the composite pass does not paint the sky over them.
    auto* map = static_cast<GpuParticle*>(m_ring.begin());
    int out = 0;
    int plainEnd = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < m_live; ++i) {
            const Particle& p = m_pool[i];
            if (p.solid != (pass == 1))
                continue;
            const glm::vec3 rel(p.pos - cameraPos);
            // fade out over the last third of the life rather than popping
            const float t = p.life * p.invLife;
            const float alpha = std::min(1.0f, t * 3.0f) * p.maxAlpha;
            GpuParticle& g = map[out++];
            g.x = rel.x;
            g.y = rel.y;
            g.z = rel.z;
            g.size = p.size;
            if (p.layer < 0) {
                g.r = p.tint.r;
                g.g = p.tint.g;
                g.b = p.tint.b;
            } else {
                g.r = g.g = g.b = 1.0f; // tint comes from the block texture
            }
            g.a = alpha;
            g.layer = float(p.layer);
        }
        if (pass == 0)
            plainEnd = out;
    }

    glProgramUniformMatrix4fv(m_prog, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
    glProgramUniform3fv(m_prog, 4, 1, &right.x);
    glProgramUniform3fv(m_prog, 5, 1, &up.x);
    glProgramUniform1i(m_prog, 6, 0);
    glUseProgram(m_prog);
    glBindTextureUnit(0, blockTexArray);
    m_ring.bind(0);
    glBindVertexArray(m_vao);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE); // debris must not write depth or it fights the water pass
    if (plainEnd > 0)
        glDrawArrays(GL_TRIANGLES, 0, plainEnd * 6);
    if (out > plainEnd) {
        glDepthMask(GL_TRUE);
        glDrawArrays(GL_TRIANGLES, plainEnd * 6, (out - plainEnd) * 6);
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    m_ring.fence();
}

} // namespace vx
