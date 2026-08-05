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

Chunk* LightEngine::lookupChunk(ChunkPos p) const {
    ChunkCacheEntry& e = m_chunkCache[cacheSlot(p)];
    if (!e.valid || !(e.key == p)) {
        e.key = p;
        e.value = m_world.chunkDataMutable(p);
        e.valid = true;
    }
    return e.value;
}

void LightEngine::resetChunkCache() {
    for (ChunkCacheEntry& e : m_chunkCache)
        e.valid = false;
}

BlockId LightEngine::blockAt(const glm::ivec3& wp) const {
    if (wp.y >= kWorldTop)
        return blocks::Air;
    if (wp.y < 0)
        return blocks::Stone;
    const Chunk* c = lookupChunk(chunkOf(wp));
    if (!c)
        return blocks::Stone; // unloaded = opaque; light stops at the loaded edge
    return c->get(wp.x & 31, wp.y & 31, wp.z & 31);
}

int LightEngine::getLevel(const glm::ivec3& wp, int channel) const {
    if (wp.y >= kWorldTop)
        return channel == Sky ? 15 : 0;
    if (wp.y < 0)
        return 0;
    const Chunk* c = lookupChunk(chunkOf(wp));
    if (!c)
        return 0;
    const std::uint16_t v = c->lightAt(wp.x & 31, wp.y & 31, wp.z & 31);
    return channel == Sky ? light::sky(v) : light::block(v);
}

void LightEngine::setLevel(const glm::ivec3& wp, int channel, int level) {
    if (wp.y < 0 || wp.y >= kWorldTop)
        return;
    Chunk* c = lookupChunk(chunkOf(wp));
    if (!c)
        return;
    const int x = wp.x & 31;
    const int y = wp.y & 31;
    const int z = wp.z & 31;
    const std::uint16_t old = c->lightAt(x, y, z);
    // block light writes all three colour channels: the engine is still monochrome, and
    // keeping them equal means light::block() is exact and output matches the old path
    const std::uint16_t next =
        channel == Sky ? light::withSky(old, level) : light::withBlock(old, level);
    if (next == old)
        return;
    c->setLight(x, y, z, next);
    markTouched(wp);
}

void LightEngine::markTouched(const glm::ivec3& wp) {
    const auto add = [this](ChunkPos p) {
        if (p.y < 0 || p.y >= kWorldChunksY)
            return;
        if (m_touchedSet.insert(p).second)
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

bool LightEngine::seedStackValid(Chunk** stack) const {
    // Refetched every step, never cached across a yield: the column can be evicted or
    // have a chunk swapped under us (the coarse->detailed re-gen double buffer) between
    // frames, and a stale pointer here would be a use-after-free.
    for (int cy = 0; cy < kWorldChunksY; ++cy) {
        stack[cy] = m_world.chunkDataMutable({m_seed.chunkX, cy, m_seed.chunkZ});
        if (!stack[cy])
            return false;
    }
    return true;
}

void LightEngine::beginSeedColumn(std::int32_t chunkX, std::int32_t chunkZ) {
    resetChunkCache();
    m_seedAborted = false;
    m_seed.chunkX = chunkX;
    m_seed.chunkZ = chunkZ;
    m_seed.cursor = 0;
    m_seed.phase = SeedPhase::Descent;

    Chunk* stack[kWorldChunksY] = {};
    if (!seedStackValid(stack)) {
        m_seed.phase = SeedPhase::Idle;
        m_seedAborted = true; // nothing was lit; the caller has to try again later
        return;
    }
    // whole-chunk fast path: with heightmap-driven terrain a uniform-air chunk sits
    // entirely above the surface, so it is all direct sky. cheap, so it isn't split.
    for (int cy = 0; cy < kWorldChunksY; ++cy)
        if (stack[cy]->uniform() && stack[cy]->uniformId() == blocks::Air)
            stack[cy]->fillLight(light::kFullSky);
}

bool LightEngine::stepSeedColumn(double budgetMs) {
    resetChunkCache();
    if (m_seed.phase == SeedPhase::Idle)
        return true;

    Chunk* stack[kWorldChunksY] = {};
    if (!seedStackValid(stack)) {
        // column went away mid-seed. the partial work is dropped, so this is NOT a
        // finished column even though there is nothing further to step.
        m_seed.phase = SeedPhase::Idle;
        m_seedAborted = true;
        return true;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto over = [&] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                   .count() > budgetMs;
    };
    const int baseX = m_seed.chunkX * kChunkSize;
    const int baseZ = m_seed.chunkZ * kChunkSize;
    constexpr int kCells = kChunkSize * kChunkSize;
    // How often to read the clock, per phase, because the phases are wildly different
    // costs per cell. Descent is a straight pointer walk. Spread can climb 256 voxels
    // doing six world lookups each, so it needs a much tighter leash — checking it as
    // rarely as the descent let the 2 ms budget run to 8-10 ms. (Those lookups go through
    // the chunk cache now and are far cheaper, but the ratio between the phases holds.)
    constexpr int kDescentCheck = 32;
    constexpr int kSpreadCheck = 4;

    // straight-down descent, direct pointers so this stays cheap (no hashing)
    if (m_seed.phase == SeedPhase::Descent) {
        while (m_seed.cursor < kCells) {
            if (m_seed.cursor % kDescentCheck == 0 && m_seed.cursor > 0 && over())
                return false;
            const int dz = m_seed.cursor >> 5;
            const int dx = m_seed.cursor & 31;
            int y = kWorldTop - 1;
            for (; y >= 0; --y) {
                Chunk* c = stack[y >> 5];
                if (c->uniform() && c->uniformId() == blocks::Air) {
                    y = y & ~31; // whole chunk already filled above; skip it
                    continue;
                }
                if (blocks::opaque(c->get(dx, y & 31, dz)))
                    break;
                c->setLight(dx, y & 31, dz,
                            light::withSky(c->lightAt(dx, y & 31, dz), 15));
            }
            m_seed.heights[dz][dx] = y;
            ++m_seed.cursor;
        }
        m_seed.cursor = 0;
        m_seed.phase = SeedPhase::Spread;
    }

    // spread sources: direct-sky voxels bordering shadowed air (cave mouths, overhangs).
    // pure heightmap terrain has none — the loop breaks immediately per column
    if (m_seed.phase == SeedPhase::Spread) {
        while (m_seed.cursor < kCells) {
            if (m_seed.cursor % kSpreadCheck == 0 && m_seed.cursor > 0 && over())
                return false;
            const int dz = m_seed.cursor >> 5;
            const int dx = m_seed.cursor & 31;
            for (int sy = m_seed.heights[dz][dx] + 1; sy < kWorldTop; ++sy) {
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
            ++m_seed.cursor;
        }
        m_seed.cursor = 0;
        m_seed.phase = SeedPhase::Emissive;
    }

    // emissive blocks — palette check skips the scan for almost every chunk
    while (m_seed.cursor < kWorldChunksY) {
        if (m_seed.cursor > 0 && over())
            return false;
        const int cy = m_seed.cursor;
        ++m_seed.cursor;
        const Chunk* c = stack[cy];
        if (!c->paletteHas(blocks::Glow))
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

    m_seed.phase = SeedPhase::Idle;
    return true;
}

void LightEngine::seedColumn(std::int32_t chunkX, std::int32_t chunkZ) {
    beginSeedColumn(chunkX, chunkZ);
    while (!stepSeedColumn(1e9)) {
        // budget is effectively infinite: run it out in one go
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
    resetChunkCache();
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
    resetChunkCache();
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

std::uint16_t LightEngine::lightAt(const glm::ivec3& wp) const {
    // debug/verification only, and callers reach it at arbitrary times — never trust a
    // cache filled by whatever ran last
    const_cast<LightEngine*>(this)->resetChunkCache();
    const int b = getLevel(wp, Block);
    return light::pack(b, b, b, getLevel(wp, Sky));
}

} // namespace vx
