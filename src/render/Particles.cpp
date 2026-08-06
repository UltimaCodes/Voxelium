#include "render/Particles.hpp"

#include "core/Log.hpp"
#include "gfx/Shader.hpp"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>

namespace vx {

namespace {
constexpr GLbitfield kWriteMapFlags =
    GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
constexpr float kGravity = 14.0f; // lighter than the player's 32: debris should hang a bit
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
    glCreateBuffers(1, &m_ssbo);
    glNamedBufferStorage(m_ssbo, GLsizeiptr(kCapacity) * GLsizeiptr(sizeof(GpuParticle)),
                         nullptr, kWriteMapFlags);
    m_map = static_cast<GpuParticle*>(glMapNamedBufferRange(
        m_ssbo, 0, GLsizeiptr(kCapacity) * GLsizeiptr(sizeof(GpuParticle)), kWriteMapFlags));
    if (!m_map) {
        VX_ERROR("particles: persistent mapping failed");
        return false;
    }
    return true;
}

void Particles::shutdown() {
    if (m_ssbo)
        glDeleteBuffers(1, &m_ssbo);
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    m_ssbo = m_vao = 0;
    m_map = nullptr;
    m_live = 0;
}

Particles::Particle& Particles::acquire() {
    if (m_live < kCapacity)
        return m_pool[m_live++];
    // Full: overwrite slot 0. The pool is compacted on update, so slot 0 is the oldest
    // survivor — dropping that is far better than dropping the burst being spawned now.
    return m_pool[0];
}

void Particles::spawnBlockBreak(const glm::dvec3& blockMin, int texLayer) {
    for (int i = 0; i < 14; ++i) {
        Particle& p = acquire();
        p.pos = blockMin + glm::dvec3(m_rand01(), m_rand01(), m_rand01());
        p.vel = glm::vec3(m_rand01() - 0.5f, m_rand01() * 0.9f + 0.15f, m_rand01() - 0.5f) *
                4.2f;
        p.life = 0.5f + m_rand01() * 0.45f;
        p.invLife = 1.0f / p.life;
        p.size = 0.055f + m_rand01() * 0.055f;
        p.layer = texLayer;
    }
}

void Particles::spawnSplash(const glm::dvec3& at) {
    for (int i = 0; i < 10; ++i) {
        Particle& p = acquire();
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

void Particles::spawnFootstep(const glm::dvec3& at, int texLayer) {
    for (int i = 0; i < 4; ++i) {
        Particle& p = acquire();
        p.pos = at + glm::dvec3(m_rand01() * 0.4 - 0.2, 0.04, m_rand01() * 0.4 - 0.2);
        p.vel = glm::vec3((m_rand01() - 0.5f) * 1.1f, m_rand01() * 0.7f,
                          (m_rand01() - 0.5f) * 1.1f);
        p.life = 0.28f + m_rand01() * 0.22f;
        p.invLife = 1.0f / p.life;
        p.size = 0.05f + m_rand01() * 0.04f;
        p.layer = texLayer;
    }
}

void Particles::update(float dt) {
    // Swap-remove into a compacted prefix: the draw call then submits [0, m_live) with no
    // gaps and no per-particle liveness test in the shader.
    for (int i = 0; i < m_live;) {
        Particle& p = m_pool[i];
        p.life -= dt;
        if (p.life <= 0.0f) {
            m_pool[i] = m_pool[--m_live];
            continue;
        }
        p.vel.y -= kGravity * dt;
        p.pos += glm::dvec3(p.vel) * double(dt);
        ++i;
    }
}

void Particles::draw(const glm::mat4& viewProj, const glm::dvec3& cameraPos,
                     const glm::vec3& right, const glm::vec3& up,
                     unsigned int blockTexArray) {
    if (m_live == 0)
        return;
    for (int i = 0; i < m_live; ++i) {
        const Particle& p = m_pool[i];
        const glm::vec3 rel(p.pos - cameraPos);
        // fade out over the last third of the life rather than popping
        const float t = p.life * p.invLife;
        const float alpha = std::min(1.0f, t * 3.0f);
        GpuParticle& g = m_map[i];
        g.x = rel.x;
        g.y = rel.y;
        g.z = rel.z;
        g.size = p.size;
        if (p.layer < 0) {
            g.r = 0.62f;
            g.g = 0.80f;
            g.b = 0.95f;
        } else {
            g.r = g.g = g.b = 1.0f; // tint comes from the block texture
        }
        g.a = alpha;
        g.layer = float(p.layer);
    }

    glProgramUniformMatrix4fv(m_prog, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
    glProgramUniform3fv(m_prog, 4, 1, &right.x);
    glProgramUniform3fv(m_prog, 5, 1, &up.x);
    glProgramUniform1i(m_prog, 6, 0);
    glUseProgram(m_prog);
    glBindTextureUnit(0, blockTexArray);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo);
    glBindVertexArray(m_vao);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE); // debris must not write depth or it fights the water pass
    glDrawArrays(GL_TRIANGLES, 0, m_live * 6);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

} // namespace vx
