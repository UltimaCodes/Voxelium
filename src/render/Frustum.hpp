#pragma once

#include <glm/glm.hpp>

namespace vx {

// Frustum planes pulled out of a view-projection matrix (Gribb/Hartmann). The main-view
// perspective frustum uses 5 planes (the infinite reversed-Z projection has no far
// plane, clip.z is a positive constant that always passes); shadow cascades are ortho
// boxes and use all 6. `count` tells the cull shader how many to test.
struct Frustum {
    glm::vec4 planes[6];
    int count = 5;

    static Frustum fromViewProj(const glm::mat4& m) {
        const auto row = [&](int i) {
            return glm::vec4(m[0][i], m[1][i], m[2][i], m[3][i]);
        };
        Frustum f;
        f.count = 5;
        f.planes[0] = row(3) + row(0); // left
        f.planes[1] = row(3) - row(0); // right
        f.planes[2] = row(3) + row(1); // bottom
        f.planes[3] = row(3) - row(1); // top
        f.planes[4] = row(3) - row(2); // near (0..1 clip depth: z <= w)
        return f;
    }

    // All 6 planes for an ortho cascade so shadow casters outside the cascade box are
    // culled instead of drawn in every cascade.
    static Frustum fromOrtho(const glm::mat4& m) {
        const auto row = [&](int i) {
            return glm::vec4(m[0][i], m[1][i], m[2][i], m[3][i]);
        };
        Frustum f;
        f.count = 6;
        f.planes[0] = row(3) + row(0);
        f.planes[1] = row(3) - row(0);
        f.planes[2] = row(3) + row(1);
        f.planes[3] = row(3) - row(1);
        f.planes[4] = row(3) + row(2); // near (0..1 clip)
        f.planes[5] = row(3) - row(2); // far
        return f;
    }

    // Positive-vertex test; planes don't need normalizing for a sign check.
    bool intersects(const glm::vec3& center, const glm::vec3& extent) const {
        for (int i = 0; i < count; ++i) {
            const glm::vec3 n(planes[i]);
            if (glm::dot(n, center) + planes[i].w + glm::dot(glm::abs(n), extent) < 0.0f)
                return false; // box entirely behind this plane
        }
        return true;
    }
};

} // namespace vx
