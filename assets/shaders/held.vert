#version 460 core

// The block in your hand: a unit cube (or a plant's two crossed planes) built from gl_VertexID,
// drawn last over the finished frame with its own projection, so it never clips into the world.
layout(location = 0) uniform mat4 uProj;
layout(location = 1) uniform mat4 uModel;
layout(location = 2) uniform uint uBlock;
layout(location = 3) uniform int uCross;

out vec3 vUvl;  // uv, then texture layer
out float vShade;

#include "texlayer.glsl"

// faces of a unit box: outward normal and two tangents, u x v = n so the winding is CCW
const vec3 kN[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0),
                           vec3(0, 0, 1), vec3(0, 0, -1));
const vec3 kU[6] = vec3[6](vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 0, 1), vec3(1, 0, 0),
                           vec3(1, 0, 0), vec3(0, 1, 0));
const vec3 kV[6] = vec3[6](vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0), vec3(0, 0, 1),
                           vec3(0, 1, 0), vec3(1, 0, 0));
const vec2 kCorner[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
// the light a face catches, as the world's faces do
const float kFace[6] = float[6](0.78, 0.62, 1.0, 0.5, 0.88, 0.7);

void main() {
    int face = gl_VertexID / 6;
    vec2 cc = kCorner[gl_VertexID % 6];
    vec3 p;
    vec2 uv;
    uint dir;
    if (uCross != 0) {
        // two diagonal planes, each drawn from both sides (faces 0-3)
        float plane = float(face & 1) * 2.0 - 1.0;
        vec2 c = (face & 2) == 0 ? cc : vec2(-cc.x, cc.y);
        p = vec3(c.x * 0.5, c.y * 0.5, c.x * 0.5 * plane);
        uv = vec2(0.5 + 0.5 * c.x, 0.5 - 0.5 * c.y);
        dir = 6u;
        vShade = 1.0;
    } else {
        vec3 n = kN[face], u = kU[face], v = kV[face];
        p = (n + u * cc.x + v * cc.y) * 0.5;
        dir = uint(face);
        if (face == 2 || face == 3)
            uv = vec2(0.5 + p.x, 0.5 + p.z);
        else
            uv = vec2(0.5 + (abs(n.x) > 0.5 ? p.z : p.x) * sign(n.x + n.z), 0.5 - p.y);
        vShade = kFace[face];
    }
    vUvl = vec3(uv, float(texLayer(uBlock, dir)));
    gl_Position = uProj * uModel * vec4(p, 1.0);
}
