#include "io/WorldStore.hpp"

#include "core/Log.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>

namespace vx {
namespace {

constexpr const char* kMeta = "world.cfg";

// Copies the .vxr region files only. Anything else in a world folder is ours (the meta
// file) or is not ours to move around.
int copyRegions(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::error_code ec;
    std::filesystem::create_directories(to, ec);
    if (!std::filesystem::exists(from, ec))
        return 0;
    int n = 0;
    for (const auto& e : std::filesystem::directory_iterator(from, ec)) {
        if (e.path().extension() != ".vxr")
            continue;
        std::filesystem::copy_file(e.path(), to / e.path().filename(),
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (!ec)
            ++n;
    }
    return n;
}

void removeRegions(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec))
        return;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
        if (e.path().extension() == ".vxr")
            std::filesystem::remove(e.path(), ec);
}

std::size_t dirBytes(const std::filesystem::path& dir) {
    std::error_code ec;
    std::size_t total = 0;
    if (!std::filesystem::exists(dir, ec))
        return 0;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
        if (e.is_regular_file(ec))
            total += std::size_t(e.file_size(ec));
    return total;
}

} // namespace

WorldStore::WorldStore(std::filesystem::path root) : m_root(std::move(root)) {
    std::error_code ec;
    std::filesystem::create_directories(m_root / "worlds", ec);
}

std::string WorldStore::sanitize(const std::string& name) {
    std::string out;
    for (const char c : name) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (std::isalnum(u) || c == '-' || c == '_' || c == ' ')
            out.push_back(c);
    }
    // leading/trailing spaces make folders that are miserable to deal with by hand
    while (!out.empty() && out.front() == ' ')
        out.erase(out.begin());
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    if (out.size() > 48)
        out.resize(48);
    return out;
}

void WorldStore::clearLive() const {
    removeRegions(liveDir());
    std::error_code ec;
    std::filesystem::create_directories(liveDir(), ec);
}

std::vector<WorldStore::Entry> WorldStore::list() const {
    std::vector<Entry> out;
    std::error_code ec;
    const auto dir = m_root / "worlds";
    if (!std::filesystem::exists(dir, ec))
        return out;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_directory(ec))
            continue;
        Entry w;
        w.name = e.path().filename().string();
        std::ifstream in(e.path() / kMeta);
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("seed=", 0) != 0)
                continue;
            const std::string v = line.substr(5);
            std::from_chars(v.data(), v.data() + v.size(), w.seed);
        }
        w.bytes = dirBytes(e.path());
        out.push_back(std::move(w));
    }
    std::sort(out.begin(), out.end(),
              [](const Entry& a, const Entry& b) { return a.name < b.name; });
    return out;
}

bool WorldStore::save(const std::string& name, int seed) const {
    const std::string clean = sanitize(name);
    if (clean.empty())
        return false;
    const auto dir = worldDir(clean);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
        return false;
    // Stale regions from a previous save under this name would otherwise survive as chunks
    // the new save never had.
    removeRegions(dir);
    {
        std::ofstream out(dir / kMeta, std::ios::trunc);
        if (!out)
            return false;
        out << "seed=" << seed << "\n";
    }
    const int n = copyRegions(liveDir(), dir);
    VX_INFO("world saved: \"%s\" seed %d, %d region file(s)", clean.c_str(), seed, n);
    return true;
}

bool WorldStore::load(const std::string& name, int& seedOut) const {
    const auto dir = worldDir(sanitize(name));
    std::error_code ec;
    if (!std::filesystem::exists(dir / kMeta, ec))
        return false;
    bool got = false;
    std::ifstream in(dir / kMeta);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("seed=", 0) != 0)
            continue;
        const std::string v = line.substr(5);
        if (std::from_chars(v.data(), v.data() + v.size(), seedOut).ec == std::errc())
            got = true;
    }
    if (!got)
        return false;
    clearLive();
    const int n = copyRegions(dir, liveDir());
    VX_INFO("world loaded: \"%s\" seed %d, %d region file(s)", name.c_str(), seedOut, n);
    return true;
}

bool WorldStore::remove(const std::string& name) const {
    const std::string clean = sanitize(name);
    if (clean.empty())
        return false;
    std::error_code ec;
    const auto n = std::filesystem::remove_all(worldDir(clean), ec);
    return !ec && n > 0;
}

} // namespace vx
