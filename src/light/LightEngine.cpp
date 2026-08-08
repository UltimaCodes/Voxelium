#include "light/LightEngine.hpp"

#include "world/ChunkManager.hpp"

#include <algorithm>
#include <chrono>

namespace vx {

namespace {

// There is no world ceiling any more, so these are properties of the TERRAIN: the
// generator never writes a solid voxel above kTerrainTop or below kTerrainBottom. Anything
// out there that is not loaded is therefore open sky above and bedrock below — which is
// what lets an unloaded neighbour be answered without a world bound.
constexpr int kSkyAbove = kTerrainTop;
constexpr int kRockBelow = kTerrainBottom;

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
    if (wp.y < kRockBelow)
        return blocks::Stone;
    const Chunk* c = lookupChunk(chunkOf(wp));
    if (c)
        return c->get(wp.x & 31, wp.y & 31, wp.z & 31);
    // Not loaded. Above the terrain there is provably nothing, so it is open air; anywhere
    // else, treat it as opaque so light stops at the loaded edge instead of leaking.
    return wp.y >= kSkyAbove ? blocks::Air : blocks::Stone;
}

int LightEngine::getLevel(const glm::ivec3& wp, int channel) const {
    if (wp.y < kRockBelow)
        return 0;
    const Chunk* c = lookupChunk(chunkOf(wp));
    if (!c)
        return (channel == Sky && wp.y >= kSkyAbove) ? 15 : 0;
    const std::uint16_t v = c->lightAt(wp.x & 31, wp.y & 31, wp.z & 31);
    switch (channel) {
        case R: return light::red(v);
        case G: return light::green(v);
        case B: return light::blue(v);
        default: return light::sky(v);
    }
}

void LightEngine::setLevel(const glm::ivec3& wp, int channel, int level) {
    // No world bound to test: a voxel exists wherever a chunk is loaded, and nowhere else.
    Chunk* c = lookupChunk(chunkOf(wp));
    if (!c)
        return;
    const int x = wp.x & 31;
    const int y = wp.y & 31;
    const int z = wp.z & 31;
    const std::uint16_t old = c->lightAt(x, y, z);
    const std::uint16_t next =
        channel == R   ? light::pack(level, light::green(old), light::blue(old), light::sky(old))
        : channel == G ? light::pack(light::red(old), level, light::blue(old), light::sky(old))
        : channel == B ? light::pack(light::red(old), light::green(old), level, light::sky(old))
                       : light::withSky(old, level);
    if (next == old)
        return;
    c->setLight(x, y, z, next);
    markTouched(wp);
}

void LightEngine::markTouched(const glm::ivec3& wp) {
    const auto add = [this](ChunkPos p) {
        // any chunk Y is legal now; a position with no chunk simply misses on lookup
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

bool LightEngine::seedStackValid(Chunk** stack, int& y0, int& y1, bool& skyAtTop) const {
    m_world.columnRange(m_seed.chunkX, m_seed.chunkZ, y0, y1, skyAtTop);
    if (y1 - y0 + 1 > kMaxStack)
        return false; // range wider than the band should ever produce; refuse rather than overrun
    // Refetched every step, never cached across a yield: the column can be evicted or
    // have a chunk swapped under us (the coarse->detailed re-gen double buffer) between
    // frames, and a stale pointer here would be a use-after-free.
    for (int cy = y0; cy <= y1; ++cy) {
        stack[cy - y0] = m_world.chunkDataMutable({m_seed.chunkX, cy, m_seed.chunkZ});
        if (!stack[cy - y0])
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

    Chunk* stack[kMaxStack] = {};
    int y0 = 0, y1 = 0;
    bool skyAtTop = false;
    if (!seedStackValid(stack, y0, y1, skyAtTop)) {
        m_seed.phase = SeedPhase::Idle;
        m_seedAborted = true; // nothing was lit; the caller has to try again later
        return;
    }
    m_seed.y0 = y0;
    m_seed.y1 = y1;
    m_seed.skyAtTop = skyAtTop;
    // whole-chunk fast path: a uniform-air chunk above the surface is all direct sky. Only
    // valid when the column actually reaches the sky — buried air is dark, not lit.
    if (skyAtTop)
        for (int cy = y0; cy <= y1; ++cy)
            if (stack[cy - y0]->uniform() && stack[cy - y0]->uniformId() == blocks::Air)
                stack[cy - y0]->fillLight(light::kFullSky);
}

bool LightEngine::stepSeedColumn(double budgetMs) {
    resetChunkCache();
    if (m_seed.phase == SeedPhase::Idle)
        return true;

    Chunk* stack[kMaxStack] = {};
    int y0 = 0, y1 = 0;
    bool skyAtTop = false;
    if (!seedStackValid(stack, y0, y1, skyAtTop) || y0 != m_seed.y0 || y1 != m_seed.y1) {
        // column went away mid-seed, or the band moved under us. the partial work is
        // dropped, so this is NOT a finished column even though there is nothing to step.
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
            const int topY = (m_seed.y1 + 1) * kChunkSize - 1;
            const int botY = m_seed.y0 * kChunkSize;
            int y = topY;
            // Buried band: sky attenuates 1 per step and a chunk of separation is 32, so
            // nothing above can reach here. Record the whole column as blocked and skip.
            if (!m_seed.skyAtTop) {
                m_seed.heights[dz][dx] = topY;
                ++m_seed.cursor;
                continue;
            }
            for (; y >= botY; --y) {
                Chunk* c = stack[(y >> 5) - m_seed.y0];
                if (c->uniform() && c->uniformId() == blocks::Air) {
                    y = y & ~31; // whole chunk already filled above; skip it
                    continue;
                }
                // The direct-sky column stops at anything that is not clear air. A cutout
                // (leaves) does not BLOCK light, but it does not pass it at full strength
                // either — so the descent ends here and the BFS carries it down from this
                // voxel, attenuating per step. That is what makes a canopy dapple instead
                // of either going black or letting a full-strength shaft through.
                const BlockId b = c->get(dx, y & 31, dz);
                if (blocks::blocksLight(b) || blocks::lightCost(b) > 1)
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
            // Up to the top of the resident range, not to a world ceiling — the spread
            // only ever climbs from the surface it found during the descent anyway.
            const int spreadTop = (m_seed.y1 + 1) * kChunkSize;
            for (int sy = m_seed.heights[dz][dx] + 1; sy < spreadTop; ++sy) {
                const glm::ivec3 p{baseX + dx, sy, baseZ + dz};
                bool boundary = false;
                for (int d = 0; d < 6 && !boundary; ++d) {
                    const glm::ivec3 n = p + kStep[d];
                    boundary = !blocks::blocksLight(blockAt(n)) && getLevel(n, Sky) < 14;
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
    while (m_seed.cursor < m_seed.y1 - m_seed.y0 + 1) {
        if (m_seed.cursor > 0 && over())
            return false;
        const int slot = m_seed.cursor; // index into the band, NOT an absolute chunk Y
        const int cy = m_seed.y0 + slot;
        ++m_seed.cursor;
        const Chunk* c = stack[slot];
        if (!c->paletteHasEmissive())
            continue;
        for (int y = 0; y < kChunkSize; ++y)
            for (int z = 0; z < kChunkSize; ++z)
                for (int x = 0; x < kChunkSize; ++x) {
                    const BlockId id = c->get(x, y, z);
                    if (blocks::emissive(id)) {
                        const glm::ivec3 wp{baseX + x, cy * kChunkSize + y, baseZ + z};
                        const int e[3] = {blocks::emitR(id), blocks::emitG(id),
                                          blocks::emitB(id)};
                        for (int ch = R; ch <= B; ++ch)
                            if (e[ch] > 0) {
                                setLevel(wp, ch, e[ch]);
                                enqueue(ch, wp);
                            }
                    }
                }
    }

    m_seed.phase = SeedPhase::Idle;
    return true;
}

void LightEngine::seedNewChunk(ChunkPos pos, bool skyAbove) {
    resetChunkCache();
    Chunk* c = m_world.chunkDataMutable(pos);
    if (!c)
        return;
    const int baseX = pos.x * kChunkSize;
    const int baseY = pos.y * kChunkSize;
    const int baseZ = pos.z * kChunkSize;

    if (const Chunk* above = m_world.chunkData({pos.x, pos.y + 1, pos.z})) {
        // Take the sky from the layer overhead rather than recomputing a descent:
        // propagateStep already carries full-strength sky straight down with no
        // attenuation, so re-propagating that layer writes exactly what the descent would
        // have, and it costs nothing where the neighbour above is dark (almost everywhere
        // at these depths — the BFS dies on the first step).
        for (int z = 0; z < kChunkSize; ++z)
            for (int x = 0; x < kChunkSize; ++x)
                if (light::sky(above->lightAt(x, 0, z)) > 0)
                    enqueue(Sky, {baseX + x, baseY + kChunkSize, baseZ + z});
    } else if (skyAbove) {
        // New top of the band on a column that reaches daylight. Nothing above to descend
        // from, so this chunk IS the sky — the generator never writes solid above
        // kTerrainTop, which is what makes that safe to assert rather than trace.
        for (int y = 0; y < kChunkSize; ++y)
            for (int z = 0; z < kChunkSize; ++z)
                for (int x = 0; x < kChunkSize; ++x) {
                    if (blocks::blocksLight(c->get(x, y, z)))
                        continue;
                    c->setLight(x, y, z, light::withSky(c->lightAt(x, y, z), 15));
                    enqueue(Sky, {baseX + x, baseY + y, baseZ + z});
                }
    }

    // Emitters inside it. Natural cave lamps live at exactly the depths the band's bottom
    // sweeps through, so skipping this leaves lit caves dark.
    if (c->paletteHasEmissive())
        for (int y = 0; y < kChunkSize; ++y)
            for (int z = 0; z < kChunkSize; ++z)
                for (int x = 0; x < kChunkSize; ++x) {
                    const BlockId id = c->get(x, y, z);
                    if (!blocks::emissive(id))
                        continue;
                    const glm::ivec3 wp{baseX + x, baseY + y, baseZ + z};
                    const int e[3] = {blocks::emitR(id), blocks::emitG(id), blocks::emitB(id)};
                    for (int ch = R; ch <= B; ++ch)
                        if (e[ch] > 0) {
                            setLevel(wp, ch, e[ch]);
                            enqueue(ch, wp);
                        }
                }
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
        const BlockId nb = blockAt(n);
        if (blocks::blocksLight(nb))
            continue;
        // sky light at full strength keeps full strength going straight down...
        int att = (channel == Sky && d == 3 && level == 15) ? 0 : 1;
        // ...unless it is crossing a cutout, which costs extra either way. Without this a
        // shaft would pass through a canopy undimmed and leaves would light nothing.
        att += blocks::lightCost(nb) - 1;
        const int target = level - att;
        if (getLevel(n, channel) >= target)
            continue;
        setLevel(n, channel, target);
        enqueue(channel, n);
    }
}

void LightEngine::removeStep(int channel, Removal r) {
    for (int d = 0; d < 6; ++d) {
        const glm::ivec3 n = r.pos + kStep[d];
        if (blocks::blocksLight(blockAt(n)))
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
    const int oldE[3] = {blocks::emitR(oldId), blocks::emitG(oldId), blocks::emitB(oldId)};
    const int newE[3] = {blocks::emitR(newId), blocks::emitG(newId), blocks::emitB(newId)};

    if (blocks::blocksLight(newId) && !blocks::blocksLight(oldId)) {
        // placed a solid block: kill whatever light was in this voxel
        for (int ch = 0; ch < kChannels; ++ch) {
            const int level = getLevel(wp, ch);
            if (level > 0) {
                setLevel(wp, ch, 0);
                m_removal[ch].push_back({wp, std::uint8_t(level)});
            }
        }
    }
    if (!blocks::blocksLight(newId) && blocks::blocksLight(oldId)) {
        // broke a block: pull light in from every neighbor, and if the voxel above has
        // direct sky, the column below regains it
        for (int d = 0; d < 6; ++d)
            for (int ch = 0; ch < kChannels; ++ch)
                enqueue(ch, wp + kStep[d]);
        if (getLevel(wp + glm::ivec3(0, 1, 0), Sky) == 15) {
            for (int y = wp.y; y >= 0 && !blocks::blocksLight(blockAt({wp.x, y, wp.z})); --y) {
                setLevel({wp.x, y, wp.z}, Sky, 15);
                enqueue(Sky, {wp.x, y, wp.z});
            }
        }
        markTouched(wp);
    }
    for (int ch = R; ch <= B; ++ch) {
        if (oldE[ch] == newE[ch])
            continue; // this channel is unchanged; touching it would only churn the BFS
        // Retract first, using the level this voxel ACTUALLY holds rather than the old
        // emitter's nominal value. Swapping a bright emitter for a dimmer one used to take
        // the "just seed the new value" path, and seeding only ever propagates — it never
        // pulls back what the old one already lit, so the bright pool stayed and the new
        // colour bled into it. Unreachable while Glow was the only emitter; the lamps make
        // emitter-to-emitter a normal move.
        const int cur = getLevel(wp, ch);
        if (cur > 0) {
            setLevel(wp, ch, 0);
            m_removal[ch].push_back({wp, std::uint8_t(cur)});
        }
        if (newE[ch] > 0) {
            setLevel(wp, ch, newE[ch]);
            enqueue(ch, wp);
        }
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
        for (int ch = 0; ch < kChannels; ++ch) {
            for (int n = 0; n < 128 && m_removalHead[ch] < m_removal[ch].size(); ++n) {
                removeStep(ch, m_removal[ch][m_removalHead[ch]++]);
                didWork = true;
            }
            if (m_removalHead[ch] >= m_removal[ch].size()) {
                m_removal[ch].clear();
                m_removalHead[ch] = 0;
            }
        }
        for (int ch = 0; ch < kChannels; ++ch) {
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
    return light::pack(getLevel(wp, R), getLevel(wp, G), getLevel(wp, B),
                       getLevel(wp, Sky));
}

} // namespace vx
