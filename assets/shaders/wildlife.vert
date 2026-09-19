#version 460 core

// One camera-facing quad per creature, vertex-pulled like the particles. The difference is
// the heading: a fish or a gull has to point where it is going, so each sprite carries a 2D
// direction in the camera's own (right, up) plane and the quad is turned to match. The shape
// itself is drawn in the fragment shader in unrotated sprite space, so nothing here knows
// what a butterfly looks like.
struct S {
    vec4 posSize; // camera-relative centre, then half-extent
    vec4 tint;    // rgb (HDR, so a firefly can exceed 1), then alpha
    vec4 kindDir; // x = kind, yz = heading as (cos, sin), w = animation phase
};
layout(std430, binding = 0) readonly buffer Sprites { S sprites[]; };

layout(location = 0) uniform mat4 uViewProj; // 0..3
layout(location = 4) uniform vec3 uRight;
layout(location = 5) uniform vec3 uUp;

out vec2 vP;    // sprite space, -1..1
out vec4 vTint;
flat out vec2 vKindPhase;

const vec2 kQuad[6] =
    vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    S s = sprites[gl_VertexID / 6];
    vec2 c = kQuad[gl_VertexID % 6];
    vec2 h = normalize(s.kindDir.yz + vec2(1e-5, 0.0));
    vec2 r = vec2(c.x * h.x - c.y * h.y, c.x * h.y + c.y * h.x);
    vec3 world = s.posSize.xyz + (uRight * r.x + uUp * r.y) * s.posSize.w;
    vP = c;
    vTint = s.tint;
    vKindPhase = vec2(s.kindDir.x, s.kindDir.w);
    gl_Position = uViewProj * vec4(world, 1.0);
}
