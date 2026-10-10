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

// swimming: buoyant, draggy, hold jump to rise. tuned so you bob and sink slowly rather
// than plummeting like the old "water isn't there" behaviour.
constexpr double kSwimSpeed = 0.6;   // horizontal scale vs walking
constexpr double kSwimAccel = 8.0;
constexpr double kWaterGravity = 8.0; // much gentler than air -> slow sink
constexpr double kSwimUpAccel = 30.0; // holding jump overcomes gravity and rises
constexpr double kSwimRise = 4.0;     // clamp vertical speed both ways
constexpr double kSwimSink = 3.0;
constexpr double kWaterVDrag = 0.9; // per-tick vertical drag -> settles to a bob

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

    inWater = submergedAt(world);
    if (inWater) {
        const double a = std::min(kSwimAccel * kTickDt, 1.0);
        velocity.x += (wish.x * kSwimSpeed - velocity.x) * a;
        velocity.z += (wish.z * kSwimSpeed - velocity.z) * a;
        velocity.y -= kWaterGravity * kTickDt;
        if (input.jump)
            velocity.y += kSwimUpAccel * kTickDt;
        velocity.y = std::clamp(velocity.y * kWaterVDrag, -kSwimSink, kSwimRise);
    } else {
        const double accel = (onGround ? kGroundAccel : kAirAccel) * kTickDt;
        velocity.x += (wish.x - velocity.x) * std::min(accel, 1.0);
        velocity.z += (wish.z - velocity.z) * std::min(accel, 1.0);
        if (input.jump && onGround)
            velocity.y = kJumpVelocity;
        velocity.y = std::max(velocity.y - kGravity * kTickDt, -kTerminal);
    }

    onGround = false;
    moveAxis(world, 1, velocity.y * kTickDt); // y first so ground state is fresh
    moveAxis(world, 0, velocity.x * kTickDt);
    moveAxis(world, 2, velocity.z * kTickDt);
}

void Player::nudge(const ChunkManager& world, const glm::dvec2& d) {
    if (d.x != 0.0)
        moveAxis(world, 0, d.x);
    if (d.y != 0.0)
        moveAxis(world, 2, d.y);
}

// chest-deep test: water at the AABB centre means we're swimming, not wading
bool Player::submergedAt(const ChunkManager& world) const {
    const glm::dvec3 c = position + glm::dvec3(0.0, kHeight * 0.5, 0.0);
    const glm::ivec3 p{int(std::floor(c.x)), int(std::floor(c.y)), int(std::floor(c.z))};
    return blocks::isWater(world.blockAtWorld(p));
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
