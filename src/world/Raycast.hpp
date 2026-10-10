#pragma once

#include "world/Chunk.hpp"

#include <glm/glm.hpp>

#include <cmath>

namespace vx {

struct RaycastHit {
    bool hit = false;
    glm::ivec3 voxel{0};  // the solid voxel that was hit
    glm::ivec3 normal{0}; // face normal; voxel + normal is where a block would go
    double distance = 0.0; // along the ray to where it entered that voxel
};

// Amanatides & Woo voxel DDA. `solid` is any callable ivec3 -> bool.
template <typename SolidFn>
RaycastHit raycastVoxels(const glm::dvec3& origin, const glm::vec3& dir, double maxDist,
                         SolidFn&& solid) {
    glm::ivec3 voxel{int(std::floor(origin.x)), int(std::floor(origin.y)),
                     int(std::floor(origin.z))};
    glm::ivec3 step{0};
    glm::dvec3 tMax{0.0};
    glm::dvec3 tDelta{0.0};
    for (int a = 0; a < 3; ++a) {
        const double d = double(dir[a]);
        if (d > 1e-9) {
            step[a] = 1;
            tDelta[a] = 1.0 / d;
            tMax[a] = (double(voxel[a] + 1) - origin[a]) / d;
        } else if (d < -1e-9) {
            step[a] = -1;
            tDelta[a] = -1.0 / d;
            tMax[a] = (double(voxel[a]) - origin[a]) / d;
        } else {
            tDelta[a] = 1e30;
            tMax[a] = 1e30;
        }
    }

    RaycastHit result;
    double t = 0.0;
    glm::ivec3 lastStepAxisNormal{0};
    while (t <= maxDist) {
        if (solid(voxel)) {
            result.hit = true;
            result.voxel = voxel;
            result.normal = lastStepAxisNormal;
            result.distance = t;
            return result;
        }
        const int axis = tMax.x < tMax.y ? (tMax.x < tMax.z ? 0 : 2)
                                         : (tMax.y < tMax.z ? 1 : 2);
        t = tMax[axis];
        tMax[axis] += tDelta[axis];
        voxel[axis] += step[axis];
        lastStepAxisNormal = {0, 0, 0};
        lastStepAxisNormal[axis] = -step[axis];
    }
    return result;
}

} // namespace vx
