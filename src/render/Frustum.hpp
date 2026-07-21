#pragma once

#include <glm/glm.hpp>

namespace vx {

// Frustum planes pulled out of a view-projection matrix (Gribb/Hartmann). Only 5
// planes: the infinite reversed-Z projection has no far plane — clip.z is a positive
// constant, so every point passes that test by construction.
struct Frustum {
    glm::vec4 planes[5];

    static Frustum fromViewProj(const glm::mat4& m) {
        const auto row = [&](int i) {
            return glm::vec4(m[0][i], m[1][i], m[2][i], m[3][i]);
        };
        Frustum f;
        f.planes[0] = row(3) + row(0); // left
        f.planes[1] = row(3) - row(0); // right
        f.planes[2] = row(3) + row(1); // bottom
        f.planes[3] = row(3) - row(1); // top
        f.planes[4] = row(3) - row(2); // near (0..1 clip depth: z <= w)
        return f;
    }

    // Positive-vertex test; planes don't need normalizing for a sign check.
    bool intersects(const glm::vec3& center, const glm::vec3& extent) const {
        for (const glm::vec4& p : planes) {
            const glm::vec3 n(p);
            if (glm::dot(n, center) + p.w + glm::dot(glm::abs(n), extent) < 0.0f)
                return false; // box entirely behind this plane
        }
        return true;
    }
};

} // namespace vx
