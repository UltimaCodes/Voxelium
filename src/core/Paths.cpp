#include "core/Paths.hpp"

namespace fs = std::filesystem;

namespace vx {

static fs::path searchUp(fs::path dir) {
    std::error_code ec;
    for (int depth = 0; depth < 6 && !dir.empty(); ++depth) {
        fs::path candidate = dir / "assets";
        if (fs::is_directory(candidate, ec))
            return candidate;
        fs::path parent = dir.parent_path();
        if (parent == dir) // hit the filesystem root
            break;
        dir = parent;
    }
    return {};
}

fs::path findAssetsDir(const char* argv0) {
    std::error_code ec;
    fs::path exe = fs::absolute(argv0, ec);
    if (!ec) {
        fs::path found = searchUp(exe.parent_path());
        if (!found.empty())
            return found;
    }
    fs::path cwd = fs::current_path(ec);
    if (ec)
        return {};
    return searchUp(cwd);
}

} // namespace vx
