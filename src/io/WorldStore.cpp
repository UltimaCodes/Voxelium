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
// file) or is not ours to move around. -1 if any of them would not copy.
int copyRegions(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::error_code ec;
    std::filesystem::create_directories(to, ec);
    if (!std::filesystem::exists(from, ec))
        return 0;
    int n = 0;
    bool failed = false;
    for (const auto& e : std::filesystem::directory_iterator(from, ec)) {
        if (e.path().extension() != ".vxr")
            continue;
        std::error_code cec;
        std::filesystem::copy_file(e.path(), to / e.path().filename(),
                                   std::filesystem::copy_options::overwrite_existing, cec);
        if (cec) {
            VX_ERROR("could not copy %s: %s", e.path().string().c_str(), cec.message().c_str());
            failed = true;
        } else {
            ++n;
        }
    }
    return failed || ec ? -1 : n;
}

// Every region in `dir`, or only those `keep` has no copy of.
void removeRegions(const std::filesystem::path& dir, const std::filesystem::path* keep = nullptr) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec))
        return;
    std::vector<std::filesystem::path> doomed;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
        if (e.path().extension() == ".vxr" && !(keep && std::filesystem::exists(*keep / e.path().filename(), ec)))
            doomed.push_back(e.path());
    for (const auto& p : doomed)
        std::filesystem::remove(p, ec);
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
            if (line.rfind("seed=", 0) == 0)
                std::from_chars(line.data() + 5, line.data() + line.size(), w.seed);
            else if (line.rfind("gen=", 0) == 0)
                std::from_chars(line.data() + 4, line.data() + line.size(), w.gen);
        }
        w.bytes = dirBytes(e.path());
        out.push_back(std::move(w));
    }
    std::sort(out.begin(), out.end(),
              [](const Entry& a, const Entry& b) { return a.name < b.name; });
    return out;
}

bool WorldStore::save(const std::string& name, int seed, int gen) const {
    const std::string clean = sanitize(name);
    if (clean.empty())
        return false;
    const auto dir = worldDir(clean);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
        return false;
    // Copied over the old save first, so a copy that fails part way leaves it whole rather than
    // emptied. Then the regions the new save never had go, or they would survive as its chunks.
    const int n = copyRegions(liveDir(), dir);
    if (n < 0)
        return false;
    const std::filesystem::path live = liveDir();
    removeRegions(dir, &live);
    {
        std::ofstream out(dir / kMeta, std::ios::trunc);
        out << "seed=" << seed << "\n";
        out << "gen=" << gen << "\n";
        out.close();
        if (out.fail())
            return false;
    }
    VX_INFO("world saved: \"%s\" seed %d, generator %d, %d region file(s)", clean.c_str(), seed, gen, n);
    return true;
}

bool WorldStore::load(const std::string& name, int& seedOut, int& genOut) const {
    const auto dir = worldDir(sanitize(name));
    std::error_code ec;
    if (!std::filesystem::exists(dir / kMeta, ec))
        return false;
    bool got = false;
    int gen = 1; // no gen= line: saved before there were versions, so the original terrain
    std::ifstream in(dir / kMeta);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("seed=", 0) == 0) {
            if (std::from_chars(line.data() + 5, line.data() + line.size(), seedOut).ec == std::errc())
                got = true;
        } else if (line.rfind("gen=", 0) == 0) {
            std::from_chars(line.data() + 4, line.data() + line.size(), gen);
        }
    }
    if (!got)
        return false;
    genOut = gen;
    clearLive();
    const int n = copyRegions(dir, liveDir());
    if (n < 0) { // half a world is worse than none: its edits would come back with holes in
        clearLive();
        return false;
    }
    VX_INFO("world loaded: \"%s\" seed %d, generator %d, %d region file(s)", name.c_str(), seedOut,
            genOut, n);
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
