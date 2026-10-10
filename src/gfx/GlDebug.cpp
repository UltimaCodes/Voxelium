#include "gfx/GlDebug.hpp"

#include "core/Log.hpp"

#include <glad/gl.h>

namespace vx {

static const char* sourceStr(GLenum source) {
    switch (source) {
        case GL_DEBUG_SOURCE_API: return "api";
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "window";
        case GL_DEBUG_SOURCE_SHADER_COMPILER: return "shader";
        case GL_DEBUG_SOURCE_THIRD_PARTY: return "3rdparty";
        case GL_DEBUG_SOURCE_APPLICATION: return "app";
        default: return "other";
    }
}

static const char* typeStr(GLenum type) {
    switch (type) {
        case GL_DEBUG_TYPE_ERROR: return "error";
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "deprecated";
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "undefined";
        case GL_DEBUG_TYPE_PORTABILITY: return "portability";
        case GL_DEBUG_TYPE_PERFORMANCE: return "perf";
        default: return "other";
    }
}

static void GLAD_API_PTR debugCb(GLenum source, GLenum type, GLuint id, GLenum severity,
                                 GLsizei /*length*/, const GLchar* message,
                                 const void* /*userParam*/) {
    // Notifications are driver chatter (buffer placement info etc.), drop them.
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
        return;
    VX_ERROR("GL[%s/%s] id=%u: %s", sourceStr(source), typeStr(type), id, message);
}

void installGlDebugCallback() {
    glEnable(GL_DEBUG_OUTPUT);
#ifndef NDEBUG
    // Synchronous means the callback fires on the offending call's stack, a breakpoint
    // in debugCb lands directly on the broken GL call instead of some later flush.
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif
    glDebugMessageCallback(debugCb, nullptr);
}

} // namespace vx
