#include "gfx/Shader.hpp"

#include "core/Log.hpp"

#include <ankerl/unordered_dense.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace vx {

namespace {

std::filesystem::path g_cacheDir;
bool g_cacheOn = false;
std::chrono::steady_clock::time_point g_start;
std::uint64_t g_glIdentity = 0; // vendor/renderer/version, so a driver update misses
int g_hits = 0;
int g_misses = 0;

// GL_ARB_parallel_shader_compile. Not in this glad build, so the entry point is fetched
// by name, it only ever raises a thread-count ceiling, so missing it is harmless.
using PFNMaxShaderCompilerThreads = void(APIENTRY*)(GLuint);
constexpr GLenum kMaxShaderCompilerThreadsArb = 0x91B0;

std::uint64_t hashBytes(const void* p, std::size_t n) {
    return ankerl::unordered_dense::detail::wyhash::hash(p, n);
}

std::uint64_t mixHash(std::uint64_t a, std::uint64_t b) {
    return ankerl::unordered_dense::detail::wyhash::mix(a ^ 0x9e3779b97f4a7c15ull, b);
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        VX_ERROR("failed to open %s", path.string().c_str());
        return {};
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// Minimal `#include "name.glsl"` for shaders, resolved relative to the including file.
// GLSL has none, and the alternative is keeping the same function in two shaders and
// watching them drift, which for a density field sampled by both the sky and the terrain
// would show up as cloud shadows that do not match the clouds.
//
// Expanded BEFORE the source is hashed for the program cache, so editing an included file
// invalidates every program that pulls it in, with no extra bookkeeping.
std::string readShaderSource(const std::filesystem::path& path, int depth = 0) {
    const std::string src = readTextFile(path);
    if (src.find("#include") == std::string::npos)
        return src;
    if (depth > 4) {
        VX_ERROR("shader includes nested too deep at %s", path.string().c_str());
        return src;
    }
    std::string out;
    out.reserve(src.size() * 2);
    std::istringstream in(src);
    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find("#include");
        const auto q1 = line.find('"');
        const auto q2 = q1 == std::string::npos ? std::string::npos : line.find('"', q1 + 1);
        if (hash == std::string::npos || q1 == std::string::npos || q2 == std::string::npos ||
            hash > q1) {
            out += line;
            out += '\n';
            continue;
        }
        const std::string name = line.substr(q1 + 1, q2 - q1 - 1);
        out += readShaderSource(path.parent_path() / name, depth + 1);
        out += '\n';
    }
    return out;
}

// Splices `defines` in after the `#version` line. GLSL requires #version to be the first
// non-comment token, so a define cannot simply be prepended.
//
// The injected block carries a #line directive so compiler errors keep reporting the line
// numbers of the ORIGINAL file, without it every message in an injected shader is off by
// the number of lines added, which is the sort of small lie that costs an hour.
std::string injectDefines(const std::string& src, std::string_view defines) {
    if (defines.empty())
        return src;
    const auto versionPos = src.find("#version");
    if (versionPos == std::string::npos)
        return std::string(defines) + src; // no #version: not our problem, but do not lose it
    const auto eol = src.find('\n', versionPos);
    if (eol == std::string::npos)
        return src;

    // count the lines consumed up to and including #version so #line restores the count
    const int versionLine = 1 + int(std::count(src.begin(), src.begin() + eol, '\n'));
    std::string out;
    out.reserve(src.size() + defines.size() + 32);
    out.append(src, 0, eol + 1);
    out.append(defines);
    if (!defines.empty() && defines.back() != '\n')
        out.push_back('\n');
    out.append("#line ").append(std::to_string(versionLine + 1)).push_back('\n');
    out.append(src, eol + 1, std::string::npos);
    return out;
}

std::filesystem::path cachePath(std::uint64_t key) {
    char name[32];
    std::snprintf(name, sizeof(name), "%016llx.bin", static_cast<unsigned long long>(key));
    return g_cacheDir / name;
}

// Cached binaries are keyed by content, so a stale file is a miss and never a wrong hit.
// The header is re-checked on read anyway: a hash collision that got past the filename
// would still have to match both halves here.
struct CacheHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint64_t key;
    std::uint64_t glIdentity;
    std::uint32_t binaryFormat;
    std::uint32_t size;
};
constexpr std::uint32_t kCacheMagic = 0x43535856; // 'VXSC'
constexpr std::uint32_t kCacheVersion = 1;

bool tryLoadBinary(std::uint64_t key, GLuint program) {
    if (!g_cacheOn)
        return false;
    std::ifstream f(cachePath(key), std::ios::binary);
    if (!f)
        return false;
    CacheHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || h.magic != kCacheMagic || h.version != kCacheVersion || h.key != key ||
        h.glIdentity != g_glIdentity || h.size == 0)
        return false;
    std::vector<char> blob(h.size);
    f.read(blob.data(), std::streamsize(h.size));
    if (!f)
        return false;

    glProgramBinary(program, GLenum(h.binaryFormat), blob.data(), GLsizei(h.size));
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    // A driver is entitled to reject its own binary (it does after some updates). That is
    // not an error, it just means compiling from source instead.
    return ok == GL_TRUE;
}

void storeBinary(std::uint64_t key, GLuint program) {
    if (!g_cacheOn)
        return;
    GLint len = 0;
    glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, &len);
    if (len <= 0)
        return;
    std::vector<char> blob(static_cast<std::size_t>(len), '\0');
    GLenum format = 0;
    GLsizei written = 0;
    glGetProgramBinary(program, len, &written, &format, blob.data());
    if (written <= 0)
        return;

    std::error_code ec;
    std::filesystem::create_directories(g_cacheDir, ec);
    std::ofstream f(cachePath(key), std::ios::binary | std::ios::trunc);
    if (!f)
        return;
    const CacheHeader h{kCacheMagic, kCacheVersion, key, g_glIdentity, std::uint32_t(format),
                        std::uint32_t(written)};
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(blob.data(), written);
}

GLuint compileStage(GLenum stage, const std::string& source, const char* label) {
    GLuint shader = glCreateShader(stage);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char infoLog[4096];
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
        VX_ERROR("shader compile failed (%s):\n%s", label, infoLog);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Shared tail: restore from cache if we can, otherwise compile every stage and link.
// Never queries link status, that is resolveProgram's job, and querying here is exactly
// what serializes the driver's compiler threads.
ProgramJob buildProgram(std::uint64_t key, std::string label,
                        std::initializer_list<std::pair<GLenum, const std::string*>> stages) {
    ProgramJob job;
    job.label = std::move(label);
    job.program = glCreateProgram();

    if (tryLoadBinary(key, job.program)) {
        ++g_hits;
        return job; // key stays 0: nothing to write back
    }
    ++g_misses;

    std::vector<GLuint> compiled;
    for (const auto& [stage, src] : stages) {
        const GLuint s = compileStage(stage, *src, job.label.c_str());
        if (!s) {
            for (const GLuint c : compiled)
                glDeleteShader(c);
            glDeleteProgram(job.program);
            job.program = 0;
            return job;
        }
        compiled.push_back(s);
        glAttachShader(job.program, s);
    }
    if (g_cacheOn)
        glProgramParameteri(job.program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
    glLinkProgram(job.program);
    for (const GLuint c : compiled)
        glDeleteShader(c); // flagged for deletion; freed when the program goes
    job.key = key;
    return job;
}

} // namespace

void initShaderCache(const std::filesystem::path& dir) {
    g_start = std::chrono::steady_clock::now();
    const auto str = [](GLenum e) {
        const char* s = reinterpret_cast<const char*>(glGetString(e));
        return std::string(s ? s : "");
    };
    const std::string identity = str(GL_VENDOR) + '|' + str(GL_RENDERER) + '|' + str(GL_VERSION);
    g_glIdentity = hashBytes(identity.data(), identity.size());

    GLint formats = 0;
    glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &formats);
    if (formats <= 0) {
        VX_INFO("shader cache: driver exposes no program binary formats, compiling from source");
    } else {
        g_cacheDir = dir;
        g_cacheOn = true;
    }

    if (auto fn = reinterpret_cast<PFNMaxShaderCompilerThreads>(
            glfwGetProcAddress("glMaxShaderCompilerThreadsARB"))) {
        fn(0xFFFFFFFFu); // no ceiling: use as many compiler threads as the driver has
        VX_INFO("shader cache: parallel compile enabled");
    }
}

void shutdownShaderCache() {
    if (g_hits + g_misses > 0) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - g_start)
                              .count();
        VX_INFO("shader cache: %d hit(s), %d compiled, %.0f ms", g_hits, g_misses, ms);
    }
    g_cacheOn = false;
}

ProgramJob submitProgram(const std::filesystem::path& vertPath,
                         const std::filesystem::path& fragPath, std::string_view defines) {
    const std::string vertSrc = injectDefines(readShaderSource(vertPath), defines);
    const std::string fragSrc = injectDefines(readShaderSource(fragPath), defines);
    if (vertSrc.empty() || fragSrc.empty())
        return {};

    std::uint64_t key = mixHash(hashBytes(vertSrc.data(), vertSrc.size()),
                                hashBytes(fragSrc.data(), fragSrc.size()));
    key = mixHash(key, g_glIdentity);
    return buildProgram(key, vertPath.filename().string() + " + " + fragPath.filename().string(),
                        {{GL_VERTEX_SHADER, &vertSrc}, {GL_FRAGMENT_SHADER, &fragSrc}});
}

ProgramJob submitTessProgram(const std::filesystem::path& vertPath,
                             const std::filesystem::path& tescPath,
                             const std::filesystem::path& tesePath,
                             const std::filesystem::path& fragPath, std::string_view defines) {
    const std::string vertSrc = injectDefines(readShaderSource(vertPath), defines);
    const std::string tescSrc = injectDefines(readShaderSource(tescPath), defines);
    const std::string teseSrc = injectDefines(readShaderSource(tesePath), defines);
    const std::string fragSrc = injectDefines(readShaderSource(fragPath), defines);
    if (vertSrc.empty() || tescSrc.empty() || teseSrc.empty() || fragSrc.empty())
        return {};

    // salted so it can never share a key with a two-stage program over the same vert + frag
    constexpr std::uint64_t kTessSalt = 0x5be0cd19137e2179ull;
    std::uint64_t key = mixHash(hashBytes(vertSrc.data(), vertSrc.size()),
                                hashBytes(tescSrc.data(), tescSrc.size()));
    key = mixHash(key, hashBytes(teseSrc.data(), teseSrc.size()));
    key = mixHash(key, hashBytes(fragSrc.data(), fragSrc.size()));
    key = mixHash(key, mixHash(g_glIdentity, kTessSalt));
    return buildProgram(key, vertPath.filename().string() + " + " + tesePath.filename().string() +
                                 " + " + fragPath.filename().string(),
                        {{GL_VERTEX_SHADER, &vertSrc},
                         {GL_TESS_CONTROL_SHADER, &tescSrc},
                         {GL_TESS_EVALUATION_SHADER, &teseSrc},
                         {GL_FRAGMENT_SHADER, &fragSrc}});
}

ProgramJob submitComputeProgram(const std::filesystem::path& compPath,
                                std::string_view defines) {
    const std::string src = injectDefines(readShaderSource(compPath), defines);
    if (src.empty())
        return {};

    // salted differently from the vert+frag key so a compute shader can never collide
    // with a graphics program that happens to hash the same
    constexpr std::uint64_t kComputeSalt = 0xc03303f7a19b45d1ull;
    const std::uint64_t key =
        mixHash(hashBytes(src.data(), src.size()), mixHash(g_glIdentity, kComputeSalt));
    return buildProgram(key, compPath.filename().string(), {{GL_COMPUTE_SHADER, &src}});
}

unsigned int resolveProgram(ProgramJob& job) {
    if (job.program == 0)
        return 0;

    GLint ok = GL_FALSE;
    glGetProgramiv(job.program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char infoLog[4096];
        glGetProgramInfoLog(job.program, sizeof(infoLog), nullptr, infoLog);
        VX_ERROR("program link failed (%s):\n%s", job.label.c_str(), infoLog);
        glDeleteProgram(job.program);
        job.program = 0;
        return 0;
    }
    if (job.key != 0)
        storeBinary(job.key, job.program);
    return job.program;
}

unsigned int loadProgram(const std::filesystem::path& vertPath,
                         const std::filesystem::path& fragPath) {
    ProgramJob job = submitProgram(vertPath, fragPath);
    return resolveProgram(job);
}

unsigned int loadComputeProgram(const std::filesystem::path& compPath) {
    ProgramJob job = submitComputeProgram(compPath);
    return resolveProgram(job);
}

} // namespace vx
