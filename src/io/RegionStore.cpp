#include "io/RegionStore.hpp"

#include <algorithm>

#include "core/Log.hpp"
#include "io/ChunkSerializer.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <utility>

namespace vx {

namespace {

constexpr char kMagic[4] = {'V', 'X', 'R', 'G'};
// v2: sparse directory keyed by chunk Y. v1 was a dense slot grid whose length encoded
// the world height, which no longer exists as a concept.
constexpr std::uint32_t kVersion = 2;
constexpr std::size_t kHeader = 12;   // magic, version, entry count
constexpr std::size_t kDirEntry = 16; // chunkY, packed locals, offset, length

std::int32_t floorDiv(std::int32_t a, std::int32_t b) {
    const std::int32_t q = a / b;
    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(std::uint8_t(v));
    out.push_back(std::uint8_t(v >> 8));
    out.push_back(std::uint8_t(v >> 16));
    out.push_back(std::uint8_t(v >> 24));
}

std::uint32_t read32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) |
           (std::uint32_t(p[3]) << 24);
}

} // namespace

RegionStore::RegionStore(std::filesystem::path dir) : m_dir(std::move(dir)) {
    std::error_code ec;
    std::filesystem::create_directories(m_dir, ec);
}

RegionStore::RegionKey RegionStore::regionOf(ChunkPos pos) {
    return {floorDiv(pos.x, kRegionSize), floorDiv(pos.z, kRegionSize)};
}

std::int64_t RegionStore::localKey(ChunkPos pos) {
    const std::int64_t lx = pos.x - floorDiv(pos.x, kRegionSize) * kRegionSize;
    const std::int64_t lz = pos.z - floorDiv(pos.z, kRegionSize) * kRegionSize;
    return (std::int64_t(pos.y) << 8) | (lx << 4) | lz;
}

std::filesystem::path RegionStore::regionPath(RegionKey key) const {
    char name[64];
    std::snprintf(name, sizeof(name), "r.%d.%d.vxr", key.x, key.z);
    return m_dir / name;
}

RegionStore::Region& RegionStore::ensureLoaded(RegionKey key) {
    const auto it = m_regions.find(key);
    if (it != m_regions.end())
        return it->second;

    Region region;
    std::ifstream file(regionPath(key), std::ios::binary);
    if (file) {
        std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
        const bool magicOk = buf.size() >= 4 && std::memcmp(buf.data(), kMagic, 4) == 0;
        const bool versionOk = buf.size() >= 8 && read32(buf.data() + 4) == kVersion;
        if (magicOk && versionOk && buf.size() >= kHeader) {
            const std::size_t count = read32(buf.data() + 8);
            const std::size_t dataStart = kHeader + count * kDirEntry;
            if (dataStart <= buf.size()) {
                for (std::size_t i = 0; i < count; ++i) {
                    const std::uint8_t* e = buf.data() + kHeader + i * kDirEntry;
                    const std::int32_t cy = std::int32_t(read32(e));
                    const std::uint32_t locals = read32(e + 4);
                    const std::uint32_t off = read32(e + 8);
                    const std::uint32_t len = read32(e + 12);
                    if (len == 0)
                        continue;
                    const std::size_t begin = dataStart + off;
                    if (begin + len > buf.size())
                        continue;
                    const std::int64_t k =
                        (std::int64_t(cy) << 8) | std::int64_t(locals & 0xFFu);
                    region.slots[k].assign(buf.begin() + std::ptrdiff_t(begin),
                                           buf.begin() + std::ptrdiff_t(begin + len));
                }
            }
        } else if (magicOk && buf.size() >= 8) {
            VX_ERROR("region %s uses save format v%u, this build reads v%u, ignoring it "
                     "(the terrain regenerates; edits in it are lost)",
                     regionPath(key).string().c_str(), read32(buf.data() + 4), kVersion);
        } else {
            VX_ERROR("region %s is corrupt, ignoring", regionPath(key).string().c_str());
        }
    }
    return m_regions.emplace(key, std::move(region)).first->second;
}

bool RegionStore::load(ChunkPos pos, Chunk& chunk) {
    std::lock_guard lock(m_mutex);
    Region& region = ensureLoaded(regionOf(pos));
    const auto it = region.slots.find(localKey(pos));
    if (it == region.slots.end() || it->second.empty())
        return false;
    return ChunkSerializer::deserialize(it->second, chunk);
}

void RegionStore::save(ChunkPos pos, std::vector<std::uint8_t> bytes) {
    std::lock_guard lock(m_mutex);
    Region& region = ensureLoaded(regionOf(pos));
    region.slots[localKey(pos)] = std::move(bytes);
    region.dirty = true;
}

void RegionStore::writeRegion(RegionKey key, const Region& region) const {
    std::vector<std::uint8_t> out;
    out.insert(out.end(), kMagic, kMagic + 4);
    put32(out, kVersion);

    // Only non-empty slots get an entry, so the file is proportional to what was actually
    // edited rather than to the size of the addressable space.
    std::vector<std::pair<std::int64_t, const std::vector<std::uint8_t>*>> live;
    live.reserve(region.slots.size());
    for (const auto& kv : region.slots)
        if (!kv.second.empty())
            live.emplace_back(kv.first, &kv.second);
    std::sort(live.begin(), live.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    put32(out, std::uint32_t(live.size()));

    std::vector<std::uint8_t> data;
    for (const auto& [k, blob] : live) {
        put32(out, std::uint32_t(std::int32_t(k >> 8)));  // chunk Y
        put32(out, std::uint32_t(k & 0xFF));              // packed locals
        put32(out, std::uint32_t(data.size()));
        put32(out, std::uint32_t(blob->size()));
        data.insert(data.end(), blob->begin(), blob->end());
    }
    out.insert(out.end(), data.begin(), data.end());

    // write to a temp file then rename, a crash mid-write can't corrupt the region
    const std::filesystem::path path = regionPath(key);
    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec)
        VX_ERROR("failed to write region %s: %s", path.string().c_str(), ec.message().c_str());
}

void RegionStore::flush() {
    std::lock_guard lock(m_mutex);
    int written = 0;
    for (auto& [key, region] : m_regions)
        if (region.dirty) {
            writeRegion(key, region);
            region.dirty = false;
            ++written;
        }
    if (written > 0)
        VX_INFO("saved %d region(s) to %s", written, m_dir.string().c_str());
}

RegionStore::Stats RegionStore::stats() const {
    std::lock_guard lock(m_mutex);
    Stats s;
    s.regionsResident = int(m_regions.size());
    for (const auto& [key, region] : m_regions)
        for (const auto& [k, blob] : region.slots)
            if (!blob.empty()) {
                ++s.chunksStored;
                s.bytesStored += blob.size();
            }
    return s;
}

void RegionStore::setDir(std::filesystem::path dir) {
    std::lock_guard lock(m_mutex);
    m_regions.clear(); // cached regions belong to the world we are leaving
    m_dir = std::move(dir);
}

void RegionStore::discardAll() {
    std::lock_guard lock(m_mutex);
    m_regions.clear();
    std::error_code ec; // best effort: a locked file must not take the process down
    if (std::filesystem::exists(m_dir, ec))
        for (const auto& e : std::filesystem::directory_iterator(m_dir, ec))
            if (e.path().extension() == ".vxr")
                std::filesystem::remove(e.path(), ec);
}

} // namespace vx
