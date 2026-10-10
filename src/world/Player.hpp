#pragma once

#include <glm/glm.hpp>

namespace vx {

class ChunkManager;

struct PlayerInput {
    float forward = 0.0f; // -1..1 along camera yaw
    float strafe = 0.0f;
    float yawDeg = 0.0f;
    bool jump = false;
};

// Minecraft-flavored player physics on a fixed 60 Hz timestep: an AABB (0.6 x 1.8)
// swept axis-by-axis against solid voxels, gravity, jump. Runs entirely on the main
// thread; the accumulator in main keeps it deterministic no matter the framerate.
class Player {
public:
    static constexpr float kHalfWidth = 0.3f;
    static constexpr float kHeight = 1.8f;
    static constexpr float kEyeHeight = 1.62f;
    static constexpr double kTickDt = 1.0 / 60.0;

    glm::dvec3 position{0.0}; // feet center
    glm::dvec3 velocity{0.0};
    bool onGround = false;
    bool inWater = false; // chest-deep in water as of the last step(); drives swim + fog

    void step(const ChunkManager& world, const PlayerInput& input);
    // Moves the player sideways (x, z) by d, stopping at walls like a step does: the give when
    // something solid but not a block is in the way (an animal, Wildlife::interact).
    void nudge(const ChunkManager& world, const glm::dvec2& d);
    glm::dvec3 eye() const { return position + glm::dvec3(0.0, kEyeHeight, 0.0); }

private:
    bool submergedAt(const ChunkManager& world) const;
    void moveAxis(const ChunkManager& world, int axis, double delta);
};

} // namespace vx
