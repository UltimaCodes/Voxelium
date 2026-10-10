#include "light/LightEngine.hpp"

#include "world/ChunkManager.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace vx {

namespace {

// There is no world ceiling any more, so these are properties of the TERRAIN: the
// generator never writes a solid voxel above kTerrainTop or below kTerrainBottom. Anything
// out there that is not loaded is therefore open sky above and bedrock below, which is
// what lets an unloaded neighbour be answered without a world bound.
constexpr int kSkyAbove = kTerrainTop;
constexpr int kRockBelow = kTerrainBottom;

constexpr glm::ivec3 kStep[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                 {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

ChunkPos chunkOf(const glm::ivec3& wp) {
    return {wp.x >> 5, wp.y >> 5, wp.z >> 5};
}

// Chunk stores light at 8 bits/channel (0..255, see Chunk.hpp's light namespace) but BFS
// below reasons in 16 discrete steps (0..15) throughout, unchanged, propagation distance,
// neighbour comparisons, queue behaviour, all exactly as before this widening. *17 is exact
// (15*17=255) so the round trip is lossless with no rounding ambiguity. The widened range's
// real benefit shows up at MESH time, not here: GreedyMesher averages up to 4 neighbouring
// voxels per face corner, and can now land on genuine intermediate values (e.g. a corner
// between step 5 [85] and step 6 [102] can read ~94) instead of being truncated straight
// back to one of only 16 integers the instant the average is computed.
constexpr int kLevelScale = 17;
int storedFromLevel(int level) { return level * kLevelScale; }
int levelFromStored(int stored) { return stored / kLevelScale; }

// Accumulates wall time into `target` on scope exit, early return included, a seed phase's
// while loop can bail out mid-column via its own budget check, so the only exit that is
// guaranteed to run is a destructor, not a line manually placed after the loop.
class ScopedNanoAccum {
public:
    explicit ScopedNanoAccum(std::uint64_t& target) : m_target(target) {}
    ~ScopedNanoAccum() {
        m_target += std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - m_t0)
                                       .count());
    }

private:
    std::uint64_t& m_target;
    std::chrono::steady_clock::time_point m_t0 = std::chrono::steady_clock::now();
};

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
    const std::uint32_t v = c->lightAt(wp.x & 31, wp.y & 31, wp.z & 31);
    switch (channel) {
        case R: return levelFromStored(light::red(v));
        case G: return levelFromStored(light::green(v));
        case B: return levelFromStored(light::blue(v));
        default: return levelFromStored(light::sky(v));
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
    const std::uint32_t old = c->lightAt(x, y, z);
    const int stored = storedFromLevel(level);
    const std::uint32_t next =
        channel == R   ? light::pack(stored, light::green(old), light::blue(old), light::sky(old))
        : channel == G ? light::pack(light::red(old), stored, light::blue(old), light::sky(old))
        : channel == B ? light::pack(light::red(old), light::green(old), stored, light::sky(old))
                       : light::withSky(old, stored);
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
    // valid when the column actually reaches the sky, buried air is dark, not lit.
    if (skyAtTop)
        for (int cy = y0; cy <= y1; ++cy)
            if (stack[cy - y0]->uniform() && stack[cy - y0]->uniformId() == blocks::Air)
                stack[cy - y0]->fillSky(255);
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
    // How often to read the clock: every 32 cells. Descent and spread are both pointer walks
    // and queue pushes now; the spread used to do six world lookups a voxel and needed a check
    // every 4 cells to keep inside the budget.
    constexpr int kDescentCheck = 32;

    // straight-down descent, direct pointers so this stays cheap (no hashing)
    if (m_seed.phase == SeedPhase::Descent) {
        ScopedNanoAccum _timeThis(m_seedPhaseNanos.descent);
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
                // either, so the descent ends here and the BFS carries it down from this
                // voxel, attenuating per step. That is what makes a canopy dapple instead
                // of either going black or letting a full-strength shaft through.
                const BlockId b = c->get(dx, y & 31, dz);
                if (blocks::blocksLight(b) || blocks::lightCost(b) > 1)
                    break;
                c->setLight(dx, y & 31, dz,
                            light::withSky(c->lightAt(dx, y & 31, dz), 255));
            }
            m_seed.heights[dz][dx] = y;
            ++m_seed.cursor;
        }
        m_seed.cursor = 0;
        m_seed.phase = SeedPhase::Spread;
    }

    // spread sources: direct-sky voxels next to air the descent did not reach (overhangs, cave
    // mouths, the space under a canopy). Read off the descent heights, not off the light: a
    // cell's sky voxels are sources from just above where its own descent stopped up to where
    // a neighbouring cell's stopped, plus the one over a cutout the descent stopped on. This
    // used to test each voxel's neighbours' CURRENT light, climbing from the surface until one
    // was not dark, and the light BFS runs between the frames a seed is spread over, so where
    // a climb stopped, and with it whether the pocket above got any light, depended on timing.
    // Two loads of the same place disagreed by one or two levels on whole runs of voxels.
    //
    // Neighbouring columns are left to checkColumnEdges, which compares the shared faces once
    // both sides are lit. Counting an unlit neighbour (sky 0 only because nobody had descended
    // into it yet) as shade was 94% of all spread work once (90.9M of 96.9M enqueues in 35 s of
    // flying at radius 32) and the light backlog grew without end.
    if (m_seed.phase == SeedPhase::Spread) {
        ScopedNanoAccum _timeThis(m_seedPhaseNanos.spread);
        const int botY = m_seed.y0 * kChunkSize;
        const int spreadTop = (m_seed.y1 + 1) * kChunkSize;
        static constexpr int kNb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        while (m_seed.cursor < kCells) {
            if (m_seed.cursor % kDescentCheck == 0 && m_seed.cursor > 0 && over())
                return false;
            const int dz = m_seed.cursor >> 5;
            const int dx = m_seed.cursor & 31;
            const int h = m_seed.heights[dz][dx];
            int reach = h;
            for (const auto& n : kNb) {
                const int nx = dx + n[0], nz = dz + n[1];
                if (nx >= 0 && nx < kChunkSize && nz >= 0 && nz < kChunkSize)
                    reach = std::max(reach, m_seed.heights[nz][nx]);
            }
            // ...and only where light from here has somewhere to go: the voxel the descent
            // stopped on lets light through (leaves), or a neighbouring cell at this height is
            // below its own descent but open (an overhang, a cave mouth, under a canopy). Every
            // sky voxel beside a taller neighbour went in before, solid hillside or not: tens of
            // thousands of no-op entries a column on steep ground, a light backlog of tens of
            // millions while streaming, and the first block edit after it drained the lot.
            // Still read off the blocks and the descent alone, never off the light.
            const bool openBelow =
                h >= botY && !blocks::blocksLight(stack[(h >> 5) - m_seed.y0]->get(dx, h & 31, dz));
            for (int sy = h + 1; sy <= std::max(reach, h + 1) && sy < spreadTop; ++sy) {
                bool open = sy == h + 1 && openBelow;
                const Chunk* layer = stack[(sy >> 5) - m_seed.y0];
                for (const auto& n : kNb) {
                    const int nx = dx + n[0], nz = dz + n[1];
                    if (open || nx < 0 || nx >= kChunkSize || nz < 0 || nz >= kChunkSize ||
                        m_seed.heights[nz][nx] < sy)
                        continue;
                    open = !blocks::blocksLight(layer->get(nx, sy & 31, nz));
                }
                if (open)
                    enqueue(Sky, {baseX + dx, sy, baseZ + dz});
            }
            ++m_seed.cursor;
        }
        m_seed.cursor = 0;
        m_seed.phase = SeedPhase::Emissive;
    }

    // emissive blocks, palette check skips the scan for almost every chunk
    {
        ScopedNanoAccum _timeThis(m_seedPhaseNanos.emissive);
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
    }

    m_seed.phase = SeedPhase::Idle;
    return true;
}

void LightEngine::checkColumnEdges(std::int32_t cx, std::int32_t cz, int y0, int y1) {
    for (int cy = y0; cy <= y1; ++cy) {
        if (!m_world.chunkLit({cx, cy, cz}))
            continue;
        for (const int face : {0, 1, 4, 5})
            settleFace({cx, cy, cz}, face);
        // and the face above, inside the column too: a chunk regenerated in place starts from no
        // light, and what the chunk over it had been sending down is not re-sent by anything else
        settleFace({cx, cy, cz}, 2);
    }
    if (m_world.chunkLit({cx, y0, cz}))
        settleFace({cx, y0, cz}, 3);
}

void LightEngine::settleFace(ChunkPos pa, int face) {
    const ChunkPos pb{pa.x + kStep[face].x, pa.y + kStep[face].y, pa.z + kStep[face].z};
    const Chunk* a = m_world.chunkData(pa);
    const Chunk* b = m_world.chunkData(pb);
    if (!a || !b || !m_world.chunkLit(pb))
        return;
    constexpr int kLast = kChunkSize - 1;
    const int axis = face >> 1;
    const bool up = (face & 1) == 0; // b lies on the positive side of a
    const glm::ivec3 baseA = glm::ivec3(pa.x, pa.y, pa.z) * kChunkSize;
    const glm::ivec3 baseB = glm::ivec3(pb.x, pb.y, pb.z) * kChunkSize;
    // full sky keeps its strength going straight down; every other step costs at least one
    const bool aOverB = face == 3, bOverA = face == 2;
    for (int u = 0; u < kChunkSize; ++u)
        for (int v = 0; v < kChunkSize; ++v) {
            // the two voxels either side of the shared face; u and v run along it
            glm::ivec3 la{u, v, 0}, lb{u, v, 0};
            if (axis == 0) {
                la = {up ? kLast : 0, u, v};
                lb = {up ? 0 : kLast, u, v};
            } else if (axis == 1) {
                la = {u, up ? kLast : 0, v};
                lb = {u, up ? 0 : kLast, v};
            } else {
                la = {u, v, up ? kLast : 0};
                lb = {u, v, up ? 0 : kLast};
            }
            const std::uint32_t va = a->lightAt(la.x, la.y, la.z);
            const std::uint32_t vb = b->lightAt(lb.x, lb.y, lb.z);
            if (va == vb)
                continue;
            const bool aOpen = !blocks::blocksLight(a->get(la.x, la.y, la.z));
            const bool bOpen = !blocks::blocksLight(b->get(lb.x, lb.y, lb.z));
            const int ga[4] = {light::red(va) / 17, light::green(va) / 17, light::blue(va) / 17, light::sky(va) / 17};
            const int gb[4] = {light::red(vb) / 17, light::green(vb) / 17, light::blue(vb) / 17, light::sky(vb) / 17};
            for (int ch = 0; ch < kChannels; ++ch) {
                const int costAB = ch == Sky && aOverB && ga[ch] == 15 ? 0 : 1;
                const int costBA = ch == Sky && bOverA && gb[ch] == 15 ? 0 : 1;
                if (ga[ch] - costAB > gb[ch] && bOpen)
                    enqueue(ch, baseA + la);
                else if (gb[ch] - costBA > ga[ch] && aOpen)
                    enqueue(ch, baseB + lb);
            }
        }
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
        // at these depths, the BFS dies on the first step).
        for (int z = 0; z < kChunkSize; ++z)
            for (int x = 0; x < kChunkSize; ++x)
                if (light::sky(above->lightAt(x, 0, z)) > 0)
                    enqueue(Sky, {baseX + x, baseY + kChunkSize, baseZ + z});
    } else if (skyAbove) {
        // New top of the band on a column that reaches daylight. Nothing above to descend
        // from, so the sky comes straight in at the top face. This used to set every open voxel
        // to full sky and queue all of them: 32768 entries for a chunk of plain air, times every
        // column in view each time the camera climbed a chunk, 219M queued over one hill, far
        // faster than the light could ever be spread, and the first block edit afterwards
        // settled the lot (a freeze of seconds). Now the same descent and the same
        // only-where-it-can-go sources as the column seed; the faces are settled below.
        if (c->uniform() && c->uniformId() == blocks::Air) {
            c->fillSky(255); // all sky, nothing inside to spread
        } else {
            int stop[kChunkSize][kChunkSize]; // where each cell's descent stopped, -1 = went through
            for (int z = 0; z < kChunkSize; ++z)
                for (int x = 0; x < kChunkSize; ++x) {
                    int y = kChunkSize - 1;
                    for (; y >= 0; --y) {
                        const BlockId id = c->get(x, y, z);
                        if (blocks::blocksLight(id) || blocks::lightCost(id) > 1)
                            break;
                        c->setLight(x, y, z, light::withSky(c->lightAt(x, y, z), 255));
                    }
                    stop[z][x] = y;
                }
            static constexpr int kNb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int z = 0; z < kChunkSize; ++z)
                for (int x = 0; x < kChunkSize; ++x) {
                    const int h = stop[z][x];
                    int reach = h;
                    for (const auto& n : kNb) {
                        const int nx = x + n[0], nz = z + n[1];
                        if (nx >= 0 && nx < kChunkSize && nz >= 0 && nz < kChunkSize)
                            reach = std::max(reach, stop[nz][nx]);
                    }
                    const bool openBelow = h >= 0 && !blocks::blocksLight(c->get(x, h, z));
                    for (int y = h + 1; y <= std::max(reach, h + 1) && y < kChunkSize; ++y) {
                        bool open = y == h + 1 && openBelow;
                        for (const auto& n : kNb) {
                            const int nx = x + n[0], nz = z + n[1];
                            if (open || nx < 0 || nx >= kChunkSize || nz < 0 || nz >= kChunkSize ||
                                stop[nz][nx] < y)
                                continue;
                            open = !blocks::blocksLight(c->get(nx, y, nz));
                        }
                        if (open)
                            enqueue(Sky, {baseX + x, baseY + y, baseZ + z});
                    }
                }
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

    // and whatever the lit chunks around it would have sent in had it been here
    for (int face = 0; face < 6; ++face)
        settleFace(pos, face);
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
            // still lit from elsewhere, becomes a re-propagation source
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
        // the "just seed the new value" path, and seeding only ever propagates, it never
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

// A queue is only cleared once it has fully drained, and while the world streams in it never
// does: new work arrives every frame. The consumed front was then never given back, so the
// vector held every voxel queued since it last ran dry, 67M entries (800 MB) after half a
// minute of flying at radius 32, and each time it outgrew its capacity it copied all of that,
// which was a 33, then 68, then 132 ms frame. Dropping the consumed front once it is the bigger
// half keeps the vector near the live backlog, at O(1) per entry overall, because only the
// smaller, live part is ever moved.
template <typename T>
static void compact(std::vector<T>& q, std::size_t& head) {
    if (head < 65536 || head * 2 < q.size())
        return;
    q.erase(q.begin(), q.begin() + std::ptrdiff_t(head));
    head = 0;
}

void LightEngine::beginEdit() {
    for (int ch = 0; ch < kChannels; ++ch) {
        m_parkedQueue[ch].swap(m_queue[ch]);
        m_parkedQueueHead[ch] = std::exchange(m_queueHead[ch], 0);
        m_queue[ch].clear();
        m_parkedRemoval[ch].swap(m_removal[ch]);
        m_parkedRemovalHead[ch] = std::exchange(m_removalHead[ch], 0);
        m_removal[ch].clear();
    }
}

void LightEngine::endEdit() {
    for (int ch = 0; ch < kChannels; ++ch) {
        // whatever of the edit is still queued goes behind the parked work
        m_parkedQueue[ch].insert(m_parkedQueue[ch].end(), m_queue[ch].begin() + std::ptrdiff_t(m_queueHead[ch]),
                                 m_queue[ch].end());
        m_queue[ch].swap(m_parkedQueue[ch]);
        m_queueHead[ch] = m_parkedQueueHead[ch];
        m_parkedQueue[ch].clear();
        m_parkedRemoval[ch].insert(m_parkedRemoval[ch].end(),
                                   m_removal[ch].begin() + std::ptrdiff_t(m_removalHead[ch]), m_removal[ch].end());
        m_removal[ch].swap(m_parkedRemoval[ch]);
        m_removalHead[ch] = m_parkedRemovalHead[ch];
        m_parkedRemoval[ch].clear();
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
            } else {
                compact(m_removal[ch], m_removalHead[ch]);
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
            } else {
                compact(m_queue[ch], m_queueHead[ch]);
            }
        }
        if (!didWork)
            break;
    }
}

std::uint32_t LightEngine::lightAt(const glm::ivec3& wp) const {
    // debug/verification only, and callers reach it at arbitrary times, never trust a
    // cache filled by whatever ran last
    const_cast<LightEngine*>(this)->resetChunkCache();
    return light::pack(getLevel(wp, R), getLevel(wp, G), getLevel(wp, B),
                       getLevel(wp, Sky));
}

} // namespace vx
