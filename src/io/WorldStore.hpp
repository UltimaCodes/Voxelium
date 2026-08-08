#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vx {

// Saved worlds on disk.
//
// The world you are playing is always the scratch one in `current/`, and it is wiped at
// launch — every run starts on a fresh seed. Keeping a world is therefore a deliberate act:
// save it under a name, and it is copied into `worlds/<name>/` alongside the seed that
// generated it. Loading copies it back.
//
// Only edited chunks are ever stored (unedited terrain regenerates from the seed), so a
// saved world is the seed plus the player's diffs — usually a few tens of KB, which is why
// copying rather than renaming is affordable and keeps the live world independent of the
// snapshot it came from.
class WorldStore {
public:
    struct Entry {
        std::string name;
        int seed = 0;
        std::size_t bytes = 0;
    };

    explicit WorldStore(std::filesystem::path root);

    // Where the live world's region files go. Always the same path; contents are per-run.
    std::filesystem::path liveDir() const { return m_root / "current"; }

    // Empty the scratch world. Called once at launch, and before importing over it.
    void clearLive() const;

    std::vector<Entry> list() const;

    // Copy the live world into worlds/<name>/, recording `seed`. Overwrites a world of the
    // same name, which is what you want when re-saving one you have been playing.
    bool save(const std::string& name, int seed) const;

    // Copy worlds/<name>/ over the live world and hand back its seed. False if the world
    // is missing or has no readable seed, in which case the live world is left alone.
    bool load(const std::string& name, int& seedOut) const;

    bool remove(const std::string& name) const;

    // Filenames come from a text box, so anything that could climb out of the saves folder
    // has to go. Returns an empty string if nothing usable is left.
    static std::string sanitize(const std::string& name);

private:
    std::filesystem::path worldDir(const std::string& name) const {
        return m_root / "worlds" / name;
    }
    std::filesystem::path m_root;
};

} // namespace vx
