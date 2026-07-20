#include "gfx/Shader.hpp"

#include "core/Log.hpp"

#include <glad/gl.h>

#include <fstream>
#include <sstream>
#include <string>

namespace vx {

static std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        VX_ERROR("failed to open %s", path.string().c_str());
        return {};
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static GLuint compileStage(GLenum stage, const std::string& source, const char* label) {
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

unsigned int loadProgram(const std::filesystem::path& vertPath,
                         const std::filesystem::path& fragPath) {
    const std::string vertSrc = readTextFile(vertPath);
    const std::string fragSrc = readTextFile(fragPath);
    if (vertSrc.empty() || fragSrc.empty())
        return 0;

    GLuint vert = compileStage(GL_VERTEX_SHADER, vertSrc, vertPath.string().c_str());
    if (!vert)
        return 0;
    GLuint frag = compileStage(GL_FRAGMENT_SHADER, fragSrc, fragPath.string().c_str());
    if (!frag) {
        glDeleteShader(vert);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    glDeleteShader(vert); // flagged for deletion; freed when the program goes
    glDeleteShader(frag);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char infoLog[4096];
        glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        VX_ERROR("program link failed (%s + %s):\n%s", vertPath.string().c_str(),
                 fragPath.string().c_str(), infoLog);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

} // namespace vx
