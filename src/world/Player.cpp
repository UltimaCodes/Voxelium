#include "world/Player.hpp"

#include "world/ChunkManager.hpp"

#include <algorithm>
#include <cmath>

namespace vx {

namespace {

constexpr double kGravity = 32.0;
constexpr double kTerminal = 60.0;
constexpr double kWalkSpeed = 4.4;
constexpr double kJumpVelocity = 9.0;
constexpr double kGroundAccel = 10.0; // fraction of target velocity gained per tick-ish
constexpr double kAirAccel = 2.0;

} // namespace

void Player::step(const ChunkManager& world, const PlayerInput& input) {
    const double yaw = glm::radians(double(input.yawDeg));
    const glm::dvec3 fwd{std::sin(yaw), 0.0, -std::cos(yaw)};
    const glm::dvec3 right{fwd.z != 0.0 || fwd.x != 0.0
                               ? glm::normalize(glm::dvec3(-fwd.z, 0.0, fwd.x))
                               : glm::dvec3(1.0, 0.0, 0.0)};

    glm::dvec3 wish = fwd * double(input.forward) + right * double(input.strafe);
    if (glm::dot(wish, wish) > 1.0)
        wish = glm::normalize(wish);
    wish *= kWalkSpeed;

    const double accel = (onGround ? kGroundAccel : kAirAccel) * kTickDt;
    velocity.x += (wish.x - velocity.x) * std::min(accel, 1.0);
    velocity.z += (wish.z - velocity.z) * std::min(accel, 1.0);

    if (input.jump && onGround)
        velocity.y = kJumpVelocity;
    velocity.y = std::max(velocity.y - kGravity * kTickDt, -kTerminal);

    onGround = false;
    moveAxis(world, 1, velocity.y * kTickDt); // y first so ground state is fresh
    moveAxis(world, 0, velocity.x * kTickDt);
    moveAxis(world, 2, velocity.z * kTickDt);
}

void Player::moveAxis(const ChunkManager& world, int axis, double delta) {
    position[axis] += delta;

    const glm::dvec3 half{kHalfWidth, kHeight * 0.5, kHalfWidth};
    const glm::dvec3 center = position + glm::dvec3(0.0, kHeight * 0.5, 0.0);
    const glm::ivec3 lo{int(std::floor(center.x - half.x)), int(std::floor(center.y - half.y)),
                        int(std::floor(center.z - half.z))};
    const glm::ivec3 hi{int(std::floor(center.x + half.x)), int(std::floor(center.y + half.y)),
                        int(std::floor(center.z + half.z))};

    for (int y = lo.y; y <= hi.y; ++y)
        for (int z = lo.z; z <= hi.z; ++z)
            for (int x = lo.x; x <= hi.x; ++x) {
                if (!blocks::opaque(world.blockAtWorld({x, y, z})))
                    continue;
                constexpr double kEps = 1e-4;
                if (delta > 0.0) {
                    position[axis] = double(axis == 1 ? y : (axis == 0 ? x : z)) -
                                     (axis == 1 ? kHeight : half[axis]) - kEps;
                } else {
                    position[axis] = double((axis == 1 ? y : (axis == 0 ? x : z)) + 1) +
                                     (axis == 1 ? 0.0 : half[axis]) + kEps;
                    if (axis == 1)
                        onGround = true;
                }
                velocity[axis] = 0.0;
                return; // position clamped; no other voxel can still overlap this axis
            }
}

} // namespace vx
