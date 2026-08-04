#include "light/LightEngine.hpp"

#include "world/ChunkManager.hpp"

#include <algorithm>
#include <chrono>

namespace vx {

namespace {

constexpr int kWorldTop = kWorldChunksY * kChunkSize; // exclusive

constexpr glm::ivec3 kStep[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                 {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

ChunkPos chunkOf(const glm::ivec3& wp) {
    return {wp.x >> 5, wp.y >> 5, wp.z >> 5};
}

} // namespace

BlockId LightEngine::blockAt(const glm::ivec3& wp) const {
    if (wp.y >= kWorldTop)
        return blocks::Air;
    if (wp.y < 0)
        return blocks::Stone;
    const Chunk* c = m_world.chunkData(chunkOf(wp));
    if (!c)
        return blocks::Stone; // unloaded = opaque; light stops at the loaded edge
    return c->get(wp.x & 31, wp.y & 31, wp.z & 31);
}

int LightEngine::getLevel(const glm::ivec3& wp, int channel) const {
    if (wp.y >= kWorldTop)
        return channel == Sky ? 15 : 0;
    if (wp.y < 0)
        return 0;
    const Chunk* c = m_world.chunkData(chunkOf(wp));
    if (!c)
        return 0;
    const std::uint8_t v = c->lightAt(wp.x & 31, wp.y & 31, wp.z & 31);
    return channel == Sky ? (v >> 4) : (v & 15);
}

void LightEngine::setLevel(const glm::ivec3& wp, int channel, int level) {
    if (wp.y < 0 || wp.y >= kWorldTop)
        return;
    Chunk* c = m_world.chunkDataMutable(chunkOf(wp));
    if (!c)
        return;
    const int x = wp.x & 31;
    const int y = wp.y & 31;
    const int z = wp.z & 31;
    const std::uint8_t old = c->lightAt(x, y, z);
    const std::uint8_t next =
        channel == Sky ? std::uint8_t((old & 0x0F) | (level << 4))
                       : std::uint8_t((old & 0xF0) | level);
    if (next == old)
        return;
    c->setLight(x, y, z, next);
    markTouched(wp);
}

void LightEngine::markTouched(const glm::ivec3& wp) {
    const auto add = [this](ChunkPos p) {
        if (p.y < 0 || p.y >= kWorldChunksY)
            return;
        for (const ChunkPos& t : m_touched)
            if (t == p)
                return;
        m_touched.push_back(p);
    };
    const ChunkPos base = chunkOf(wp);
    add(base);
    // a border voxel's light is sampled by the neighbor's mesh too
    const int x = wp.x & 31;
    const int y = wp.y & 31;
    const int z = wp.z & 31;
    if (x == 0) add({base.x - 1, base.y, base.z});
    if (x == 31) add({base.x + 1, base.y, base.z});
    if (y == 0) add({base.x, base.y - 1, base.z});
    if (y == 31) add({base.x, base.y + 1, base.z});
    if (z == 0) add({base.x, base.y, base.z - 1});
    if (z == 31) add({base.x, base.y, base.z + 1});
}

void LightEngine::seedColumn(std::int32_t chunkX, std::int32_t chunkZ) {
    Chunk* stack[kWorldChunksY] = {};
    for (int cy = 0; cy < kWorldChunksY; ++cy)
        stack[cy] = m_world.chunkDataMutable({chunkX, cy, chunkZ});

    const int baseX = chunkX * kChunkSize;
    const int baseZ = chunkZ * kChunkSize;

    // whole-chunk fast path: with heightmap-driven terrain a uniform-air chunk sits
    // entirely above the surface, so it is all direct sky
    for (int cy = 0; cy < kWorldChunksY; ++cy)
        if (stack[cy] && stack[cy]->uniform() && stack[cy]->uniformId() == blocks::Air)
            stack[cy]->fillLight(0xF0);

    // straight-down descent, direct pointers so this stays cheap (no hashing)
    int heights[kChunkSize][kChunkSize];
    for (int dz = 0; dz < kChunkSize; ++dz) {
        for (int dx = 0; dx < kChunkSize; ++dx) {
            int y = kWorldTop - 1;
            for (; y >= 0; --y) {
                Chunk* c = stack[y >> 5];
                if (!c)
                    break;
                if (c->uniform() && c->uniformId() == blocks::Air) {
                    y = y & ~31; // whole chunk already filled above; skip it
                    continue;
                }
                if (blocks::opaque(c->get(dx, y & 31, dz)))
                    break;
                c->setLight(dx, y & 31, dz,
                            std::uint8_t((c->lightAt(dx, y & 31, dz) & 0x0F) | 0xF0));
            }
            heights[dz][dx] = y;
        }
    }

    // spread sources: direct-sky voxels bordering shadowed air (cave mouths, overhangs).
    // pure heightmap terrain has none — the loop breaks immediately per column
    for (int dz = 0; dz < kChunkSize; ++dz) {
        for (int dx = 0; dx < kChunkSize; ++dx) {
            for (int sy = heights[dz][dx] + 1; sy < kWorldTop; ++sy) {
                const glm::ivec3 p{baseX + dx, sy, baseZ + dz};
                bool boundary = false;
                for (int d = 0; d < 6 && !boundary; ++d) {
                    const glm::ivec3 n = p + kStep[d];
                    boundary = !blocks::opaque(blockAt(n)) && getLevel(n, Sky) < 14;
                }
                if (boundary)
                    enqueue(Sky, p);
                else
                    break;
            }
        }
    }

    // emissive blocks — palette check skips the scan for almost every chunk
    for (int cy = 0; cy < kWorldChunksY; ++cy) {
        const Chunk* c = stack[cy];
        if (!c || !c->paletteHas(blocks::Glow))
            continue;
        for (int y = 0; y < kChunkSize; ++y)
            for (int z = 0; z < kChunkSize; ++z)
                for (int x = 0; x < kChunkSize; ++x) {
                    const int lum = blocks::luminance(c->get(x, y, z));
                    if (lum > 0) {
                        const glm::ivec3 wp{baseX + x, cy * kChunkSize + y, baseZ + z};
                        setLevel(wp, Block, lum);
                        enqueue(Block, wp);
                    }
                }
    }
}

void LightEngine::propagateStep(int channel, glm::ivec3 wp) {
    const int level = getLevel(wp, channel);
    if (level <= 1)
        return;
    for (int d = 0; d < 6; ++d) {
        const glm::ivec3 n = wp + kStep[d];
        // sky light at full strength keeps full strength going straight down
        const int att = (channel == Sky && d == 3 && level == 15) ? 0 : 1;
        const int target = level - att;
        if (blocks::opaque(blockAt(n)))
            continue;
        if (getLevel(n, channel) >= target)
            continue;
        setLevel(n, channel, target);
        enqueue(channel, n);
    }
}

void LightEngine::removeStep(int channel, Removal r) {
    for (int d = 0; d < 6; ++d) {
        const glm::ivec3 n = r.pos + kStep[d];
        if (blocks::opaque(blockAt(n)))
            continue;
        const int nl = getLevel(n, channel);
        if (nl == 0)
            continue;
        const bool fedByUs =
            nl < r.oldLevel || (channel == Sky && d == 3 && nl == 15 && r.oldLevel == 15);
        if (fedByUs) {
            setLevel(n, channel, 0);
            m_removal[channel].push_back({n, std::uint8_t(nl)});
        } else {
            // still lit from elsewhere — becomes a re-propagation source
            enqueue(channel, n);
        }
    }
}

void LightEngine::onBlockChanged(const glm::ivec3& wp, BlockId oldId, BlockId newId) {
    const int oldLum = blocks::luminance(oldId);
    const int newLum = blocks::luminance(newId);

    if (blocks::opaque(newId) && !blocks::opaque(oldId)) {
        // placed a solid block: kill whatever light was in this voxel
        for (int ch = 0; ch < 2; ++ch) {
            const int level = getLevel(wp, ch);
            if (level > 0) {
                setLevel(wp, ch, 0);
                m_removal[ch].push_back({wp, std::uint8_t(level)});
            }
        }
    }
    if (!blocks::opaque(newId) && blocks::opaque(oldId)) {
        // broke a block: pull light in from every neighbor, and if the voxel above has
        // direct sky, the column below regains it
        for (int d = 0; d < 6; ++d)
            enqueue(Block, wp + kStep[d]), enqueue(Sky, wp + kStep[d]);
        if (getLevel(wp + glm::ivec3(0, 1, 0), Sky) == 15) {
            for (int y = wp.y; y >= 0 && !blocks::opaque(blockAt({wp.x, y, wp.z})); --y) {
                setLevel({wp.x, y, wp.z}, Sky, 15);
                enqueue(Sky, {wp.x, y, wp.z});
            }
        }
        markTouched(wp);
    }
    if (newLum > 0) {
        setLevel(wp, Block, newLum);
        enqueue(Block, wp);
    } else if (oldLum > 0) {
        setLevel(wp, Block, 0);
        m_removal[Block].push_back({wp, std::uint8_t(oldLum)});
    }
}

void LightEngine::process(double budgetMs) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto over = [&] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                   .count() > budgetMs;
    };

    // Slices are deliberately small: the clock is only read between them, so a slice is
    // how far past the budget this can run. Big slices made light propagation the largest
    // single overshoot in a streaming frame.
    int slice = 0;
    while (!over() || slice == 0) {
        ++slice;
        bool didWork = false;
        // removals first: re-propagation seeds they emit must see the cleared state
        for (int ch = 0; ch < 2; ++ch) {
            for (int n = 0; n < 128 && m_removalHead[ch] < m_removal[ch].size(); ++n) {
                removeStep(ch, m_removal[ch][m_removalHead[ch]++]);
                didWork = true;
            }
            if (m_removalHead[ch] >= m_removal[ch].size()) {
                m_removal[ch].clear();
                m_removalHead[ch] = 0;
            }
        }
        for (int ch = 0; ch < 2; ++ch) {
            if (!m_removal[ch].empty())
                continue; // finish removal before spreading this channel again
            for (int n = 0; n < 256 && m_queueHead[ch] < m_queue[ch].size(); ++n) {
                propagateStep(ch, m_queue[ch][m_queueHead[ch]++]);
                didWork = true;
            }
            if (m_queueHead[ch] >= m_queue[ch].size()) {
                m_queue[ch].clear();
                m_queueHead[ch] = 0;
            }
        }
        if (!didWork)
            break;
    }
}

std::uint8_t LightEngine::lightAt(const glm::ivec3& wp) const {
    return std::uint8_t((getLevel(wp, Sky) << 4) | getLevel(wp, Block));
}

} // namespace vx
