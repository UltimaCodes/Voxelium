#version 460 core

// One corner of a water quad. The quad is drawn as a 4-vertex patch (cull.comp sizes water's
// draws that way) and water.tesc / water.tese build the surface from these corners, subdividing
// the top faces near the camera so the swell can move them. Same packed unpack as chunk.vert.
layout(std430, binding = 0) readonly buffer Quads { uvec4 quads[]; };
layout(std430, binding = 1) readonly buffer Offsets { vec4 uOffsets[]; };

out vec3 vcLocal;  // position inside the chunk, at this corner's water level
out vec3 vcOffset; // the chunk's origin relative to the camera (the same for all four)
out float vcSky;   // per-corner shade, as the opaque pass has
out float vcWave;  // 0..1 how much the swell may move this corner (top faces only)
out uint vcDir;

// Corners in the order they go round the face, front side toward the viewer, which is also the
// order the tessellator's domain corners are mapped to (see water.tese).
const vec2 kCorner[6][4] = vec2[6][4](
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)),
    vec2[4](vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(0, 0)),
    vec2[4](vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0)),
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
    vec2[4](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
    vec2[4](vec2(1, 0), vec2(0, 0), vec2(0, 1), vec2(1, 1))
);

void main() {
    uvec4 q = quads[gl_VertexID / 4];
    int corner = gl_VertexID % 4;
    vec3 origin = vec3(uvec3(q.x & 31u, (q.x >> 5) & 31u, (q.x >> 10) & 31u));
    float w = float(((q.x >> 15) & 31u) + 1u);
    float h = float(((q.x >> 20) & 31u) + 1u);
    uint dir = (q.x >> 25) & 7u;

    vec2 uv = kCorner[dir][corner] * vec2(w, h);
    vec3 pos = origin;
    if (dir == 0u)      pos += vec3(1.0, uv.y, uv.x);
    else if (dir == 1u) pos += vec3(0.0, uv.y, uv.x);
    else if (dir == 2u) pos += vec3(uv.x, 1.0, uv.y);
    else if (dir == 3u) pos += vec3(uv.x, 0.0, uv.y);
    else if (dir == 4u) pos += vec3(uv.x, uv.y, 1.0);
    else                pos += vec3(uv.x, uv.y, 0.0);

    // Flowing water carries its depth in the block id (Water = full source, 14..20 are L7..L1),
    // and a shallower cell sits lower in its voxel, so a spreading sheet thins toward its edge.
    // Per-corner surface level, 4 x 3 bits in the light word's low bits (GreedyMesher::
    // waterCorners): each corner is the average depth of the columns meeting there, which is
    // what makes a thinning sheet ramp instead of stepping once per cell.
    uint corners = q.z & 0xFFFu;
    float level = float(((corners >> (uint(corner) * 3u)) & 7u) + 1u);
    // How far this cell's surface sits below the top of its voxel.
    float drop = 0.10 + (8.0 - level) * 0.105;

    // The upper edge of a side face is the same surface as the top face beside it, so it drops
    // with it (from the same per-corner level, so the wall meets the top exactly). Dropping only
    // the top face left rims of water standing proud of every shallow cell.
    bool topEdge = (dir != 2u && dir != 3u) && uv.y >= h - 0.001;
    if (dir == 2u || topEdge)
        pos.y -= drop;

    vcLocal = pos;
    vcOffset = uOffsets[gl_DrawID].xyz;
    vcDir = dir;
    // sky is the top byte of the light word (w2), scaled by this corner's shade (4 corners x 3
    // bits, whole inside w1 bits 8..19, see mesh/QuadFormat.hpp)
    uint shade12 = (q.y >> 8u) & 0xFFFu;
    float shade = float((shade12 >> (uint(corner) * 3u)) & 7u) / 7.0;
    vcSky = (float((q.z >> 24) & 255u) / 255.0) * shade;
    // wave strength, 4 x 3 bits in bits 12..23 (GreedyMesher::waveCorners); only top faces have it
    vcWave = dir == 2u ? float((q.z >> (12u + uint(corner) * 3u)) & 7u) / 7.0 : 0.0;
}
