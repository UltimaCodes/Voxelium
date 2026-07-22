#include "io/RegionStore.hpp"

#include "core/Log.hpp"
#include "io/ChunkSerializer.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <utility>

namespace vx {

namespace {

constexpr char kMagic[4] = {'V', 'X', 'R', 'G'};
constexpr std::uint32_t kVersion = 1;
constexpr int kSlots = RegionStore::kRegionSize * RegionStore::kRegionSize * kWorldChunksY;

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

int RegionStore::localIndex(ChunkPos pos) {
    const int lx = pos.x - floorDiv(pos.x, kRegionSize) * kRegionSize;
    const int lz = pos.z - floorDiv(pos.z, kRegionSize) * kRegionSize;
    const int ly = pos.y;
    return (ly * kRegionSize + lz) * kRegionSize + lx;
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
        const std::size_t headerSize = 12;
        const std::size_t dirSize = std::size_t(kSlots) * 8;
        if (buf.size() >= headerSize + dirSize && std::memcmp(buf.data(), kMagic, 4) == 0 &&
            read32(buf.data() + 4) == kVersion && int(read32(buf.data() + 8)) == kSlots) {
            const std::size_t dataStart = headerSize + dirSize;
            for (int s = 0; s < kSlots; ++s) {
                const std::uint8_t* e = buf.data() + headerSize + std::size_t(s) * 8;
                const std::uint32_t off = read32(e);
                const std::uint32_t len = read32(e + 4);
                if (len == 0)
                    continue;
                const std::size_t begin = dataStart + off;
                if (begin + len <= buf.size())
                    region.slots[std::size_t(s)].assign(buf.begin() + std::ptrdiff_t(begin),
                                                        buf.begin() +
                                                            std::ptrdiff_t(begin + len));
            }
        } else {
            VX_ERROR("region %s is corrupt or wrong version — ignoring",
                     regionPath(key).string().c_str());
        }
    }
    return m_regions.emplace(key, std::move(region)).first->second;
}

bool RegionStore::load(ChunkPos pos, Chunk& chunk) {
    std::lock_guard lock(m_mutex);
    Region& region = ensureLoaded(regionOf(pos));
    const std::vector<std::uint8_t>& blob = region.slots[std::size_t(localIndex(pos))];
    if (blob.empty())
        return false;
    return ChunkSerializer::deserialize(blob, chunk);
}

void RegionStore::save(ChunkPos pos, std::vector<std::uint8_t> bytes) {
    std::lock_guard lock(m_mutex);
    Region& region = ensureLoaded(regionOf(pos));
    region.slots[std::size_t(localIndex(pos))] = std::move(bytes);
    region.dirty = true;
}

void RegionStore::writeRegion(RegionKey key, const Region& region) const {
    std::vector<std::uint8_t> out;
    out.insert(out.end(), kMagic, kMagic + 4);
    put32(out, kVersion);
    put32(out, std::uint32_t(kSlots));

    // directory then data; offsets are relative to the start of the data section
    std::vector<std::uint8_t> data;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> dir(std::size_t(kSlots), {0, 0});
    for (int s = 0; s < kSlots; ++s) {
        const auto& blob = region.slots[std::size_t(s)];
        if (blob.empty())
            continue;
        dir[std::size_t(s)] = {std::uint32_t(data.size()), std::uint32_t(blob.size())};
        data.insert(data.end(), blob.begin(), blob.end());
    }
    for (const auto& d : dir) {
        put32(out, d.first);
        put32(out, d.second);
    }
    out.insert(out.end(), data.begin(), data.end());

    // write to a temp file then rename — a crash mid-write can't corrupt the region
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
        for (const auto& blob : region.slots)
            if (!blob.empty()) {
                ++s.chunksStored;
                s.bytesStored += blob.size();
            }
    return s;
}

} // namespace vx
